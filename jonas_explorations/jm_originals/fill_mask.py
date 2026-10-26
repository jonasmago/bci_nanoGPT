"""
Demo for a masked (bidirectional) character model: hide a few characters of a sentence and let the
model fill them in. All masked positions are predicted at once from the same corrupted input.

$ python fill_mask.py --out_dir=out-shakespeare-masked --text='To be, or not to be.' --mask_fraction=0.25
"""
import os
import pickle
import argparse
import torch
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-shakespeare-masked')
parser.add_argument('--text', default='To be, or not to be: that is the question.')
parser.add_argument('--mask_fraction', type=float, default=0.25)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
torch.manual_seed(args.seed)

# load the checkpoint and the character <-> id maps of its dataset
checkpoint = torch.load(os.path.join(args.out_dir, 'ckpt.pt'), map_location=args.device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
assert gptconf.bidirectional, "this is a causal model; use sample.py"
model = GPT(gptconf)
state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)
model.to(args.device).eval()
with open(os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# mask a random fraction of the characters
ids = torch.tensor([[stoi[c] for c in args.text]], device=args.device)
masked = torch.rand(ids.shape, device=args.device) < args.mask_fraction
x = torch.where(masked, torch.full_like(ids, gptconf.mask_token_id), ids)

with torch.no_grad():
    logits, _ = model(x)                       # [1, T, vocab]: a prediction at every position
pred = logits[..., :len(itos)].argmax(-1)      # never predict the MASK symbol itself
filled = torch.where(masked, pred, ids)

print('original:', args.text)
print('masked:  ', ''.join('_' if m else c for c, m in zip(args.text, masked[0].tolist())))
print('filled:  ', ''.join(itos[i] for i in filled[0].tolist()))
n = masked.sum().item()
print(f'{n} masked characters, {(pred[masked] == ids[masked]).sum().item()} correct')
