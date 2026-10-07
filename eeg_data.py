"""EEG token windows and masking for nanoGPT.

A dataset is a folder with manifest.json and one .npz per recording (see prepare.py). An example is a
grid [channels, frames] of codec tokens cut from one recording; channels beyond max_channels are a
random subset, fewer channels are padded (valid_mask). Causal targets are the next frame of the same
channel; masked targets are the original codes at the masked positions, -1 elsewhere.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

MIN_CHANNELS = 4  # recordings with fewer channels are skipped


def _time_mask(valid_times, ratio, span, generator):
    """Exactly ceil(ratio*N) eligible frames, with translation-unbiased spans.

    Pack requested-length spans (and at most one shorter remainder), distribute
    visible frames between them, then uniformly rotate the binary pattern. This
    avoids overshoot and endpoint clamping. A span crossing the window edge is
    represented as two edge fragments: the signal and attention never wrap.
    When few visible frames remain, adjacent spans can merge. On irregular
    validity masks, lengths count eligible frames rather than missing samples.
    """
    indices = valid_times.nonzero(as_tuple=True)[0]
    total = len(indices)
    count = max(1, math.ceil(ratio * total))
    lengths = [span] * (count // span)
    if count % span:
        lengths.append(count % span)
    device = valid_times.device
    order = torch.randperm(len(lengths), generator=generator, device=device)
    lengths = [lengths[int(i)] for i in order]
    gaps = torch.zeros(len(lengths), dtype=torch.long, device=device)
    visible = total - count
    # Keep separate spans when the available visible frames permit it.
    separated = min(visible, len(lengths))
    if separated:
        gaps[torch.randperm(len(lengths), generator=generator, device=device)[:separated]] = 1
    if visible > separated:
        slots = torch.randint(len(lengths), (visible - separated,), generator=generator, device=device)
        gaps += torch.bincount(slots, minlength=len(lengths))
    packed = torch.cat([torch.cat((torch.ones(length, dtype=torch.bool, device=device),
                                 torch.zeros(int(gap), dtype=torch.bool, device=device)))
                        for length, gap in zip(lengths, gaps)])
    shift = int(torch.randint(total, (), generator=generator, device=device))
    selected = torch.zeros_like(valid_times)
    selected[indices] = torch.roll(packed, shift)
    return selected


def make_masked_batch(tokens, valid_mask, mask_token_id, mask_ratio=0.15, mask_strategy="random",
                      mask_span=4, generator=None):
    """Replace a fraction of the tokens by MASK; targets hold the original there and -1 elsewhere.

    tokens are [B,T] (text) or [B,C,T] (EEG grid). ``random`` masks positions, ``time`` masks spans
    of frames across all channels (exactly ceil(ratio * frames), see _time_mask), ``channel`` masks
    whole channels (leaving at least one visible), ``mixed`` picks time or channel per example.
    """
    if mask_strategy not in {"random", "time", "channel", "mixed"}:
        raise ValueError(f"unknown mask_strategy {mask_strategy!r}")
    if valid_mask is None:
        valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    grid = tokens.unsqueeze(1) if tokens.ndim == 2 else tokens
    valid = valid_mask.unsqueeze(1) if tokens.ndim == 2 else valid_mask
    selected = torch.zeros_like(valid)
    for b in range(grid.size(0)):
        strategy = mask_strategy
        if strategy == "mixed":
            strategy = ("time", "channel")[int(torch.randint(2, (), generator=generator))]
        channels = valid[b].any(dim=-1).nonzero(as_tuple=True)[0]
        if strategy == "channel" and len(channels) == 1:
            strategy = "time"
        if strategy == "random":
            indices = valid[b].flatten().nonzero(as_tuple=True)[0]
            count = max(1, math.ceil(mask_ratio * len(indices)))
            order = torch.randperm(len(indices), generator=generator)[:count]
            selected[b].view(-1)[indices[order]] = True
        elif strategy == "channel":
            count = max(1, math.ceil(mask_ratio * len(channels)))
            if mask_ratio < 1:
                count = min(count, len(channels) - 1)
            order = torch.randperm(len(channels), generator=generator)[:count]
            selected[b, channels[order], :] = True
        else:
            chosen_times = _time_mask(valid[b].any(dim=0), mask_ratio, mask_span, generator)
            selected[b, :, chosen_times] = True
    selected &= valid
    selected = selected.squeeze(1) if tokens.ndim == 2 else selected
    targets = torch.full_like(tokens, -1)
    targets[selected] = tokens[selected]
    inputs = tokens.clone()
    inputs[selected] = mask_token_id
    return inputs, targets


class EEGDataset:
    """All recordings of one split in RAM (a few MB each); get_batch cuts random windows."""

    def __init__(self, data_dir, split, window_frames, max_channels):
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.data_dir / "manifest.json").read_text())
        if self.meta["tokenizer"]["n_codebooks"] != 1:
            raise ValueError("EEGDataset supports one codebook")
        self.window_frames, self.max_channels = window_frames, max_channels
        self.records = []  # per recording: codes [C,T] uint16, xyz [C,3], times [T], channel_ids [C]
        self.windows = {False: [], True: []}  # per objective: (record index, first frame, last frame + 1) of each chunk
        for record in self.meta["splits"][split]:
            with np.load(self.data_dir / record["path"]) as archive:
                data = {key: archive[key] for key in ("codes", "xyz", "times", "segment_ids", "channel_ids")}
            if data["codes"].shape[0] < MIN_CHANNELS:
                continue
            ri = len(self.records)
            self.records.append(data)
            # windows never cross a chunk boundary: chunks were normalised and tokenised independently
            segments = data["segment_ids"]
            boundaries = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, len(segments)]
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                for bidirectional in (False, True):
                    needed = window_frames + int(not bidirectional)  # causal needs one more frame for the target
                    if hi - lo >= needed:
                        self.windows[bidirectional].append((ri, int(lo), int(hi)))
        if not self.windows[True]:
            raise ValueError(f"no {split} chunks have {window_frames} frames and {MIN_CHANNELS} channels")

    def get_batch(self, batch_size, bidirectional, mask_token_id, mask_ratio, mask_strategy, mask_span, generator):
        """CPU tensors: x, y [B,C,T] and dict(xyz [B,C,3], times [B,T], channel_ids [B,C], valid_mask [B,C,T])."""
        eligible = self.windows[bidirectional]
        B, C, T = batch_size, self.max_channels, self.window_frames
        inputs = torch.zeros((B, C, T), dtype=torch.long)
        targets = torch.full_like(inputs, -1)
        valid = torch.zeros_like(inputs, dtype=torch.bool)
        xyz = torch.zeros((B, C, 3), dtype=torch.float32)
        times = torch.zeros((B, T), dtype=torch.float32)
        channel_ids = torch.zeros((B, C), dtype=torch.long)
        for b in range(B):
            ri, lo, hi = eligible[int(torch.randint(len(eligible), (), generator=generator))]
            needed = T + int(not bidirectional)
            start = int(torch.randint(lo, hi - needed + 1, (), generator=generator))
            data = self.records[ri]
            count = min(C, len(data["codes"]))
            channels = torch.randperm(len(data["codes"]), generator=generator)[:count].numpy()
            values = torch.from_numpy(data["codes"][channels, start:start + needed].astype(np.int64))
            inputs[b, :count] = values[:, :T]
            if not bidirectional:
                targets[b, :count] = values[:, 1:T + 1]
            valid[b, :count] = True
            xyz[b, :count] = torch.from_numpy(data["xyz"][channels].astype(np.float32))
            frame_times = data["times"][start:start + T]
            times[b] = torch.from_numpy((frame_times - frame_times[0]).astype(np.float32))
            channel_ids[b, :count] = torch.from_numpy(data["channel_ids"][channels].astype(np.int64))
        if bidirectional:
            inputs, targets = make_masked_batch(inputs, valid, mask_token_id, mask_ratio, mask_strategy, mask_span, generator)
        return inputs, targets, dict(xyz=xyz, times=times, channel_ids=channel_ids, valid_mask=valid)
