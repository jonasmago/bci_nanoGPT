# The one config for this repo: the original nanoGPT knobs plus the 11 EEG/masked ones (marked NEW).
# As shipped it reproduces the original "baby GPT on Shakespeare characters" run.
#   python train.py config/experiment.py          (any line can be overridden: --max_iters=100)

# ---- run / bookkeeping ----
out_dir = 'out-shakespeare-char'   # checkpoints (ckpt.pt) go here; use a new dir per experiment
eval_interval = 250                 # evaluate every N iterations
eval_iters = 200                    # batches averaged per evaluation (same fixed windows every time)
log_interval = 10                   # print the step loss every N iterations
always_save_checkpoint = False      # False: only save when val loss improves
init_from = 'scratch'               # 'scratch' | 'resume' (from out_dir) | 'gpt2*' (OpenAI weights, causal text only)
wandb_log = False                   # True: log to Weights & Biases
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

# ---- data ----
dataset = 'shakespeare_char'        # folder under data/. Text: train.bin/val.bin/meta.pkl. EEG: manifest.json + npz files
batch_size = 64                     # examples per micro-step
gradient_accumulation_steps = 1     # micro-steps per optimizer step; effective batch = batch_size * this
block_size = 256                    # attention positions. Text: tokens per sequence. EEG: >= max_channels * window_frames
window_frames = 64                  # NEW, EEG only: frames per channel per example (8 frames/s -> 64 = 8 s)
max_channels = 32                   # NEW, EEG only: channels per example; recordings with more give a random subset

# ---- model ----
n_layer = 6                         # transformer blocks
n_head = 6                          # attention heads; head_dim = n_embd / n_head
n_embd = 384                        # width of the residual stream
dropout = 0.2                       # 0 for big data; 0.1-0.2 for tiny datasets like this
bias = False                        # bias in Linear/LayerNorm (GPT-2 had True; False is slightly better)
bidirectional = False               # NEW: False = causal next-token (GPT); True = masked prediction (BERT)
position_encoding = 'learned'       # NEW: 'learned' wpe table (original) | 'rope' on the time axis (needed for EEG xyz)
channel_encoding = 'none'           # NEW, EEG: how a token knows its electrode. 'none' | 'learned' (name table) |
                                    #      'fourier' (xyz features added) | 'rope' (xyz rotations in attention, 4D RoPE) | 'rope_fourier'
rope_dims = ''                      # NEW: head dims per axis 't,x,y,z', e.g. '32,8,8,8'; '' = all to time (16 each if spatial)
use_sdpa = True                     # NEW: fused attention; False = the readable manual attention path (for pdb)

# ---- masking (bidirectional only) ----
mask_ratio = 0.15                   # NEW: fraction of positions masked (BERT 0.15; EEG 0.5)
mask_strategy = 'random'            # NEW: 'random' positions | 'time' spans across all channels | 'channel' whole channels | 'mixed'
mask_span = 4                       # NEW: frames per span for 'time'
eval_mask_strategies = 'random'     # NEW: strategies scored at evaluation (comma list), e.g. 'time,channel'

# ---- optimizer ----
learning_rate = 1e-3                # peak LR (small models tolerate higher)
max_iters = 5000                    # total optimizer steps
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99                        # 0.95 for large batches; 0.99 when tokens/step is small
grad_clip = 1.0                     # 0 disables
decay_lr = True                     # cosine schedule
warmup_iters = 100
lr_decay_iters = 5000               # usually = max_iters
min_lr = 1e-4                       # usually learning_rate / 10

# ---- system ----
device = 'cuda'                     # 'cuda', 'cuda:0', 'cpu'
dtype = 'bfloat16'                  # 'float32' | 'bfloat16' | 'float16'
compile = False                     # torch.compile: ~2x faster for causal text, but breaks pdb and is slow to start
backend = 'nccl'                    # DDP only
