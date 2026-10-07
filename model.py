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
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from position import FourierXYZ, RotaryPositions

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
        self.flash = config.use_sdpa and hasattr(F, 'scaled_dot_product_attention')
        self.bidirectional = config.bidirectional
        self.rotary = None
        if config.position_encoding == 'rope':
            self.rotary = RotaryPositions(config.n_embd // config.n_head,
                spatial=config.channel_encoding in ('rope', 'rope_fourier'), dims=config.rope_dims)

    def forward(self, x, attention_mask=None, rotary_tables=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rotary is not None:
            q, k = self.rotary.apply_rotary(q, k, rotary_tables)

        allowed = attention_mask
        causal = not self.bidirectional

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0,
                is_causal=causal and allowed is None)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if allowed is None and causal:
                allowed = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            if allowed is not None:
                att = att.masked_fill(~allowed, float('-inf'))
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
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias) #JM rescale each 384-vector to zero mean
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, rotary_tables=None):
        x = x + self.attn(self.ln_1(x), attention_mask, rotary_tables)
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
    bidirectional: bool = False # False: causal next-token (GPT). True: masked prediction (BERT)
    position_encoding: str = 'learned' # 'learned' wpe table (original) or 'rope' on the time axis
    channel_encoding: str = 'none' # EEG electrode identity: none, learned, fourier, rope, rope_fourier
    channel_vocab_size: int = 0 # size of the channel-name table for channel_encoding='learned' (UNK=0)
    mask_token_id: int = -1 # id of the MASK input symbol; -1 for causal models
    rope_dims: str = '' # head dims per axis 't,x,y,z', e.g. '32,8,8,8'; '' = automatic
    use_sdpa: bool = True # False keeps the explicit attention path for inspection

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if config.position_encoding not in ('learned', 'rope'):
            raise ValueError("position_encoding must be 'learned' or 'rope'")
        if config.channel_encoding not in ('none', 'learned', 'fourier', 'rope', 'rope_fourier'):
            raise ValueError("Unknown channel_encoding")
        if config.channel_encoding in ('rope', 'rope_fourier') and config.position_encoding != 'rope':
            raise ValueError("4D channel encoding requires position_encoding='rope'")
        if not -1 <= config.mask_token_id < config.vocab_size:
            raise ValueError("mask_token_id must be -1 or a reserved vocabulary entry")

        # Keep original module/parameter ordering for legacy AdamW checkpoints.
        modules = dict(wte=nn.Embedding(config.vocab_size, config.n_embd))
        if config.position_encoding == 'learned':
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        modules.update(drop=nn.Dropout(config.dropout),
                       h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                       ln_f=LayerNorm(config.n_embd, bias=config.bias))
        self.transformer = nn.ModuleDict(modules)
        if config.channel_encoding == 'learned':
            if config.channel_vocab_size < 1:
                raise ValueError("Learned channel encoding requires a channel vocabulary")
            self.transformer['channel_embed'] = nn.Embedding(config.channel_vocab_size, config.n_embd)
        if config.channel_encoding in ('fourier', 'rope_fourier'):
            self.transformer['xyz_embed'] = FourierXYZ(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
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
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _inputs(self, idx, xyz=None, times=None, channel_ids=None, valid_mask=None, validate=True):
        if idx.ndim not in (2, 3):
            raise ValueError("Expected token IDs [B,T] or EEG grid [B,C,T]")
        grid = idx.ndim == 3
        B, T = idx.shape[0], idx.shape[-1]
        C = idx.shape[1] if grid else 1
        L = C * T
        if T == 0 or C == 0 or L > self.config.block_size:
            raise ValueError(f"Sequence has {L} positions, budget is {self.config.block_size}")
        flat = idx.reshape(B, L)
        if valid_mask is not None:
            if valid_mask.shape != idx.shape:
                raise ValueError("valid_mask must match token IDs")
            valid_mask = valid_mask.to(device=idx.device, dtype=torch.bool).reshape(B, L)
        if times is None:
            times = torch.arange(T, device=idx.device, dtype=torch.float32).expand(B, T)
        else:
            times = times.to(device=idx.device, dtype=torch.float32)
            if times.shape == (T,):
                times = times.expand(B, T)
            if times.shape != (B, T) or (validate and not torch.isfinite(times).all()):
                raise ValueError("times must be finite [T] or [B,T] frame positions")
            if validate and T > 1 and (times[:, 1:] < times[:, :-1]).any():
                raise ValueError("Time frames must be nondecreasing")
            times = times - times[:, :1]
        spatial = self.config.channel_encoding in ('fourier', 'rope', 'rope_fourier')
        if spatial and xyz is None:
            raise ValueError("This channel_encoding requires electrode xyz in head metres")
        if xyz is None:
            xyz = torch.zeros(B, C, 3, device=idx.device)
        else:
            xyz = xyz.to(device=idx.device, dtype=torch.float32)
            if xyz.shape == (C, 3):
                xyz = xyz.expand(B, C, 3)
            if xyz.shape != (B, C, 3) or (validate and not torch.isfinite(xyz).all()):
                raise ValueError("xyz must be finite [B,C,3] head coordinates in metres")
        # Channel-major storage matches the tokenizer. Coordinates, not flattened
        # offsets, identify simultaneous frames and physical channel positions.
        t = times[:, None, :].expand(B, C, T)
        space = xyz[:, :, None, :].expand(B, C, T, 3)
        positions = torch.cat((t.unsqueeze(-1), space), dim=-1).reshape(B, L, 4)
        ids = None
        if self.config.channel_encoding == 'learned':
            if channel_ids is None or channel_ids.shape != (B, C):
                raise ValueError("Learned channel encoding requires channel_ids [B,C]")
            ids = channel_ids.to(device=idx.device, dtype=torch.long)
            if validate and ((ids < 0).any() or (ids >= self.config.channel_vocab_size).any()):
                raise ValueError("channel_ids are outside the training channel vocabulary (UNK=0)")
            ids = ids[:, :, None].expand(B, C, T).reshape(B, L)
        return flat, positions, ids, valid_mask, grid

    def encode(self, idx, *, xyz=None, times=None, channel_ids=None, valid_mask=None, validate_inputs=True):
        """Return contextual features [...,D] without constructing vocabulary logits."""
        flat, positions, ids, valid, grid = self._inputs(idx, xyz, times, channel_ids, valid_mask, validate_inputs)
        x = self.transformer.wte(flat)
        if 'wpe' in self.transformer:
            temporal = positions[..., 0]
            if validate_inputs and ((temporal != temporal.round()).any() or (temporal >= self.config.block_size).any()):
                raise ValueError("Learned positions require integer frame indices within block_size")
            x = x + self.transformer.wpe(temporal.long())
        if 'channel_embed' in self.transformer:
            x = x + self.transformer.channel_embed(ids)
        if 'xyz_embed' in self.transformer:
            # Position is constant over time within each electrode; project it once.
            B, T = idx.shape[0], idx.shape[-1]
            spatial = positions.reshape(B, -1, T, 4)[:, :, 0, 1:]
            embedding = self.transformer.xyz_embed(spatial)
            x = x + embedding[:, :, None, :].expand(-1, -1, T, -1).reshape_as(x)
        x = self.transformer.drop(x)
        if valid is not None:
            x = x * valid.unsqueeze(-1)
        # Tables and visibility are identical in every layer. Build each once.
        rotary = self.transformer.h[0].attn.rotary
        rotary_tables = rotary.prepare(positions) if rotary is not None else None
        allowed = self._attention_mask(positions, valid, grid)
        for block in self.transformer.h:
            x = block(x, allowed, rotary_tables)
        x = self.transformer.ln_f(x)
        if valid is not None:
            x = x * valid.unsqueeze(-1)
        return x.reshape(*idx.shape, self.config.n_embd)

    def _attention_mask(self, positions, valid, grid):
        length = positions.shape[1]
        allowed = None
        if not self.config.bidirectional:
            if grid:
                times = positions[..., 0]
                allowed = (times[:, :, None] >= times[:, None, :]).unsqueeze(1)
            elif valid is not None:
                allowed = torch.ones(length, length, device=positions.device, dtype=torch.bool).tril()[None, None]
        if valid is not None:
            keys = valid[:, None, None, :]
            allowed = keys if allowed is None else allowed & keys
            fallback = torch.arange(length, device=positions.device).eq(0).view(1, 1, 1, length)
            allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & fallback)
        return allowed

    def pool_features(self, idx, **kwargs):
        """Validity-aware mean pooling for a frozen downstream linear probe."""
        hidden = self.encode(idx, **kwargs).reshape(idx.shape[0], -1, self.config.n_embd)
        valid = kwargs.get('valid_mask')
        if valid is None:
            return hidden.mean(dim=1)
        count = valid.to(hidden.device).reshape(idx.shape[0], -1).sum(dim=1, keepdim=True).clamp_min(1)
        return hidden.sum(dim=1) / count

    def forward(self, idx, targets=None, *, xyz=None, times=None, channel_ids=None,
                valid_mask=None, return_all_logits=False, masked_only=False,
                score_indices=None, validate_inputs=True):
        """CPU-built score_indices avoid dynamic boolean gathers on accelerators.

        The training loader validates token IDs/coordinates before transfer and sets
        validate_inputs=False. Public calls retain value checks by default.
        """
        hidden = self.encode(idx, xyz=xyz, times=times, channel_ids=channel_ids,
                             valid_mask=valid_mask, validate_inputs=validate_inputs)
        labels = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must match token IDs")
            targets = targets.to(idx.device)
            if valid_mask is not None:
                targets = targets.masked_fill(~valid_mask.to(device=idx.device, dtype=torch.bool), -1)
            if validate_inputs and self.config.mask_token_id >= 0 and (targets == self.config.mask_token_id).any():
                raise ValueError("MASK is an input symbol, never a prediction target")
        elif masked_only or score_indices is not None:
            raise ValueError("Scored-position output requires targets")
        if masked_only:
            if score_indices is None:
                # Convenience for interactive calls; training supplies these from CPU.
                score_indices = (targets.reshape(-1) != -1).nonzero(as_tuple=True)[0]
            score_indices = score_indices.to(device=idx.device, dtype=torch.long)
            labels = targets.reshape(-1).index_select(0, score_indices)
            selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, score_indices)
        elif targets is None and not return_all_logits and not self.config.bidirectional and idx.ndim == 2:
            selected = hidden[:, [-1], :]
        else:
            selected = hidden
            if targets is not None:
                labels = targets.reshape(-1)
        logits = self.lm_head(selected)
        if self.config.mask_token_id >= 0:
            logits = logits.clone()
            logits.select(-1, self.config.mask_token_id).fill_(float('-inf'))
        loss = None
        if labels is not None:
            # Sum + tensor denominator stays finite for an entirely ignored batch
            # without a GPU-to-host truth-value check or a dynamic label gather.
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels,
                                   ignore_index=-1, reduction='sum') / (labels != -1).sum().clamp_min(1)
        return logits, loss

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Original manual attention checkpoints stored a triangular buffer. It is
        # regenerated from current times/validity now and carries no learned state.
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.endswith(('.attn.bias', '.attn.masked_bias'))}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if 'wpe' in self.transformer:
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
        if self.config.bidirectional:
            raise ValueError("generate() is for causal models; use fill_mask.py for masked text")
        if idx.ndim != 2 or self.config.channel_encoding != 'none':
            raise ValueError("generate() supports flat causal streams; EEG uses next-frame evaluation")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
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
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
