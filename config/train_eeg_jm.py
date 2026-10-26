# #jm 2026-10-19: EEG grid pretraining config = the settings of out-eeg-learned (Jonas, 2026-10-07),
# used for the channel-encoding x tokenizer comparison (jm_run_sweep.sh). Override dataset /
# channel_encoding / out_dir / max_iters on the command line.
out_dir = 'out-jm/tmp'
eval_interval = 500
eval_iters = 50
log_interval = 50
always_save_checkpoint = True

wandb_log = True
wandb_project = 'nanogpt-eeg-jm'
wandb_run_name = 'eeg'

dataset = 'eeg_dac_noPost8192'   # Chris-parameter baseline tokens (data/prepare_eeg_tokens.py default)
gradient_accumulation_steps = 4
batch_size = 4
block_size = 2048          # 32 channels x 64 frames (8 s at 8 tokens/s)
window_frames = 64
max_channels = 32

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False
bidirectional = True       # masked prediction (EEG grids require it)
mask_ratio = 0.5
mask_strategy = 'time'
mask_span = 32
channel_encoding = 'learned'  # 2026-10-23 default: one learned vector per frame (wpe) + one per channel name (wce), both added
                              # to the token, the miniFormer-style encoding; 'none' = time vector only, 'rope4d' = relative RoPE
rope_dims = '32,8,8,8'
token_input = 'codebook'   # 2026-10-23: tokens enter as their DAC codebook vectors (<dataset>/codebook.npy);
                           # 'embedding' = the original learned table per token id

learning_rate = 3e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 3e-5
warmup_iters = 100
weight_decay = 0.1
beta2 = 0.99
grad_clip = 1.0

compile = True
