"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import os # #jm: JM_PDB=1 env var stops in GPT.forward to inspect the position vectors
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from position import RotaryPositions # #jm rope4d (ported from branch eeg-reference)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = not config.bidirectional
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        if not self.flash and self.causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask=None, rope=None):
        # attn_mask: optional bool [B, 1, 1, T], True where a position may be attended to (used to hide padding)
        # rope: #jm optional (cos, sin) of shape [B, 1, T, R] from rope4d_cos_sin(); rotates the first R dims of q and k
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, query, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if rope is not None: # #jm rope4d: relative (time, x, y, z) enters through the q.k dot product
            rotary, tables = rope
            q, k = rotary.apply_rotary(q, k, tables)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal: # bidirectional models attend to every position, so no mask
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if attn_mask is not None:
                att = att.masked_fill(~attn_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, rope=None):
        x = x + self.attn(self.ln_1(x), attn_mask, rope) # #jm rope = (RotaryPositions, cos/sin tables) or None
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bidirectional: bool = False # False: causal next-token prediction (GPT). True: masked prediction (BERT), every position sees the whole sequence
    mask_token_id: int = -1 # id of the MASK input symbol for bidirectional models; -1 for causal models
    channel_encoding: str = 'none' # EEG grids: how a token knows its channel. 'none' | 'learned' (one embedding per channel name)
                                   # | 'rope4d' (#jm: axial RoPE over (frame, x, y, z) in every attention layer, position.py;
                                   #   REPLACES the wpe table, so time is relative too — the eeg-reference 'rope' design)
    channel_vocab_size: int = 0 # number of channel names for channel_encoding='learned' (id 0 = unknown)
    rope_dims: str = '32,8,8,8' # #jm rope4d only: head dims rotated by frame index, x, y, z (sum <= head size; rest unrotated)
    token_input: str = 'embedding' # #jm 2026-10-23: 'embedding' = learned table per token id (wte, the original);
                                   # 'codebook' = the token's DAC codebook VECTOR (unit-normalised, frozen buffer `codebook`)
                                   #   through a learned Linear(codebook_dim, n_embd); the MASK token gets its own learned vector.
    codebook_dim: int = 0 # #jm token_input='codebook' only: length of a codebook vector (8 for our DAC runs)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        if config.token_input == 'codebook': # #jm: tokens enter as their codebook vectors, not as ids
            assert config.codebook_dim > 0, "token_input='codebook' needs codebook_dim"
            del self.transformer['wte']
            n_codes = config.mask_token_id if config.mask_token_id >= 0 else config.vocab_size # the MASK id has no codebook row
            self.register_buffer('codebook', torch.zeros(n_codes, config.codebook_dim)) # filled by set_codebook / the checkpoint
            self.transformer['in_proj'] = nn.Linear(config.codebook_dim, config.n_embd)
            self.mask_emb = nn.Parameter(torch.zeros(config.n_embd))
        if config.channel_encoding == 'learned':
            # a second position table: wpe says WHEN a token is, wce says WHERE (which electrode); both are added to the token
            self.transformer['wce'] = nn.Embedding(config.channel_vocab_size, config.n_embd)
        if config.channel_encoding == 'rope4d': # #jm: no position tables at all; (t, x, y, z) rotate q and k in every layer
            del self.transformer['wpe']
            self.rotary = RotaryPositions(config.n_embd // config.n_head, spatial=True, dims=config.rope_dims)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if 'wte' in self.transformer: # #jm: no table to tie with when the input is the codebook vector
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        if config.token_input == 'codebook': # #jm
            torch.nn.init.normal_(self.mask_emb, mean=0.0, std=0.02)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer: # #jm rope4d has no wpe
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def set_codebook(self, codebook): # #jm: the DAC codebook [n_codes, codebook_dim]; rows unit-normalised (the quantizer's own metric)
        assert self.config.token_input == 'codebook' and tuple(codebook.shape) == tuple(self.codebook.shape), (codebook.shape, self.codebook.shape)
        self.codebook.copy_(F.normalize(codebook.float(), dim=1).to(self.codebook.device))

    def token_vectors(self, idx):
        """#jm: what the transformer receives for every token id, [..., n_embd] — the learned embedding (token_input='embedding')
        or the projected codebook vector, with the learned MASK vector where idx == mask_token_id (token_input='codebook')."""
        if self.config.token_input != 'codebook':
            return self.transformer.wte(idx)
        x = self.transformer.in_proj(self.codebook[idx.clamp(max=self.codebook.shape[0] - 1)])
        if self.config.mask_token_id >= 0:
            x = torch.where((idx == self.config.mask_token_id).unsqueeze(-1), self.mask_emb.to(x.dtype), x)
        return x

    def forward(self, idx, targets=None, channel_ids=None, valid_mask=None, xyz=None, return_hidden=False):
        # idx is either text [b, t] or an EEG grid [b, c, t] (c channels, t frames). A grid is flattened
        # channel by channel into one sequence of c*t tokens; each token keeps its frame index for wpe and,
        # with channel_encoding='learned', its channel id for wce. valid_mask [b, c, t] marks padded channels.
        # #jm xyz [b, c, 3] (head metres) feeds channel_encoding='rope4d'; return_hidden=True returns the final
        # hidden states [b, c*t, n_embd] instead of (logits, loss) (used by the classification probe).
        device = idx.device
        b, t = idx.shape[0], idx.shape[-1]
        c = idx.shape[1] if idx.ndim == 3 else 1
        assert c * t <= self.config.block_size, f"Cannot forward sequence of length {c * t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).repeat(c) # frame index of every flattened token, shape (c*t)

        # forward the GPT model itself
        x = self.token_vectors(idx.reshape(b, c * t)) # token embeddings of shape (b, c*t, n_embd)  (#jm: wte or codebook vectors)
        import pdb; pdb.set_trace()
        if 'wpe' in self.transformer: # #jm: absent for rope4d
            x = x + self.transformer.wpe(pos) # position embeddings of shape (c*t, n_embd)
        if 'wce' in self.transformer:
            x = x + self.transformer.wce(channel_ids.repeat_interleave(t, dim=1)) # channel embeddings, shape (b, c*t, n_embd)
        rope = None
        if self.config.channel_encoding == 'rope4d': # #jm: (frame, x, y, z) of every flattened token -> cos/sin tables once per forward
            assert xyz is not None, "channel_encoding='rope4d' needs xyz [b, c, 3] in head metres"
            coords = torch.cat([pos.view(1, c * t, 1).expand(b, -1, -1).float(),
                                xyz.float().repeat_interleave(t, dim=1)], dim=-1) # [b, c*t, 4]
            rope = (self.rotary, self.rotary.prepare(coords))
        x = self.transformer.drop(x)
        attn_mask = valid_mask.reshape(b, 1, 1, c * t) if valid_mask is not None else None # padded channels are never attended to
        for block in self.transformer.h:
            x = block(x, attn_mask, rope)
        x = self.transformer.ln_f(x)
        if return_hidden: # #jm
            return x

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        elif self.config.bidirectional:
            # every position is a prediction (the masked ones are the interesting ones)
            logits = self.lm_head(x).reshape(*idx.shape, -1)
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if 'wpe' in self.transformer: # #jm rope4d has no wpe
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert not self.config.bidirectional, "generate() is for causal models; use fill_mask.py for masked models"
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from thea distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
