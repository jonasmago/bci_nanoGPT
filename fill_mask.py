"""Fill masked characters with a bidirectional checkpoint (a smoke demo, not generation).

Example:
    python fill_mask.py --out_dir out-shakespeare-masked --text 'To be, or not to be.'

Predictions are simultaneous, conditioned on the same corrupted input. A supplied
--mask_positions list makes the demo reproducible without random masking.
"""

import argparse
import math
import os
import pickle

import torch

from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--text', default='To be, or not to be: that is the question.')
    parser.add_argument('--mask_fraction', type=float, default=0.25)
    parser.add_argument('--mask_positions', default='', help='Comma-separated character indices.')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data_dir', default=None, help='Override the checkpoint dataset directory.')
    args = parser.parse_args()
    if not 0 < args.mask_fraction <= 1:
        parser.error('--mask_fraction must be in (0, 1].')

    # Checkpoints are trusted local files, as in train.py and sample.py.
    checkpoint = torch.load(os.path.join(args.out_dir, 'ckpt.pt'), map_location=args.device,
                            weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    if not config.bidirectional:
        parser.error('This is a causal checkpoint; use sample.py for autoregressive generation.')
    if config.mask_token_id < 0:
        parser.error('The checkpoint has no mask_token_id.')
    saved = checkpoint.get('config', {})
    data_dir = args.data_dir or saved.get('data_dir') or os.path.join('data', saved.get('dataset', ''))
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    if 'stoi' not in meta or 'itos' not in meta:
        parser.error('This character demo requires a character dataset with stoi/itos metadata.')
    if not args.text or len(args.text) > config.block_size:
        parser.error(f'--text must contain 1 to {config.block_size} characters.')
    unknown = sorted(set(args.text) - set(meta['stoi']))
    if unknown:
        parser.error(f'Characters absent from the dataset vocabulary: {unknown!r}')

    original = torch.tensor([[meta['stoi'][c] for c in args.text]], dtype=torch.long,
                            device=args.device)
    if args.mask_positions:
        try:
            positions = sorted(set(int(i.strip()) for i in args.mask_positions.split(',')))
        except ValueError:
            parser.error('--mask_positions must be a comma-separated list of integers.')
        if any(i < 0 or i >= len(args.text) for i in positions):
            parser.error('A mask position lies outside --text.')
    else:
        generator = torch.Generator().manual_seed(args.seed)
        count = max(1, round(args.mask_fraction * len(args.text)))
        positions = torch.randperm(len(args.text), generator=generator)[:count].tolist()
    masked = original.clone()
    masked[:, positions] = config.mask_token_id
    targets = torch.full_like(original, -1)
    targets[:, positions] = original[:, positions]

    model = GPT(config).to(args.device).eval()
    state = {key.removeprefix('_orig_mod.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state)
    with torch.no_grad():
        logits, loss = model(masked, targets, masked_only=True)
        # MASK/PAD are model inputs, not printable characters.
        printable = torch.tensor(sorted(int(i) for i in meta['itos']), device=args.device)
        predictions = printable[logits[:, printable].argmax(-1)]
    completed = original.clone()
    selected = targets != -1
    completed[selected] = predictions
    display = ''.join('□' if i in positions else c for i, c in enumerate(args.text))
    filled = ''.join(meta['itos'][i] for i in completed[0].tolist())
    accuracy = (predictions == original[selected]).float().mean().item()
    print(f'Original: {args.text}')
    print(f'Masked:   {display}')
    print(f'Filled:   {filled}')
    print(f'Masked accuracy: {accuracy:.1%}; loss: {loss.item():.4f} nats; '
          f'{loss.item() / math.log(2):.4f} bits per masked token')


if __name__ == '__main__':
    main()
