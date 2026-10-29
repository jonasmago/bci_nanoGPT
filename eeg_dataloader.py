"""EEG data loader: windows from the per-recording .npz grids, never across recordings or 60 s chunks.

The grids are written by data/<dataset>/prepare.py, one file per recording in data/<dataset>/{train,val}/:
  codes [C, T] uint16, segment_ids [T] (which 60 s chunk a frame came from), channel_names [C], xyz [C, 3].

Single channel now: get_batch -> x, y [B, T], like nanoGPT's text batches (y = x shifted by one token).
Multi-channel later: get_grid_batch -> [B, C, T] plus xyz, from the same index (sketched at the bottom).
"""
import glob
import os

import numpy as np
import torch


class EEGDataLoader:
    def __init__(self, data_dir, split, block_size):
        self.recs, runs = [], []
        for f in sorted(glob.glob(os.path.join(data_dir, split, '*.npz'))):
            d = np.load(f)
            r = len(self.recs)
            self.recs.append(dict(codes=d['codes'], xyz=d['xyz'], names=d['channel_names']))
            seg = d['segment_ids']
            bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]      # 60 s chunk edges, in frames
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                n = hi - lo - block_size                                        # valid start times (x + one target token)
                if n > 0:
                    runs.append((r, lo, n, d['codes'].shape[0]))
        assert runs, f"no chunk in {data_dir}/{split} is longer than block_size={block_size} frames"
        self.runs = np.array(runs, dtype=np.int64)                              # (recording, first start, n starts, n channels)
        w = self.runs[:, 2] * self.runs[:, 3]                                   # every (channel, time) window equally likely
        self.p_single = w / w.sum()
        self.block_size = block_size
        print(f"{split}: {len(self.recs)} recordings, {len(self.runs)} chunks, {int(w.sum()):,} single-channel windows")

    def get_batch(self, batch_size):
        """Single channel: x, y [B, T] int64, y is x shifted by one token."""
        k = np.random.choice(len(self.runs), size=batch_size, p=self.p_single)
        x, y = [], []
        for r, lo, n, n_ch in self.runs[k]:
            s = lo + np.random.randint(n)                                       # start time inside the chunk
            ch = np.random.randint(n_ch)                                        # one channel of that recording
            w = self.recs[r]['codes'][ch, s:s + self.block_size + 1].astype(np.int64)
            x.append(w[:-1])
            y.append(w[1:])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))

    # Multi-channel, later (step 5):
    # def get_grid_batch(self, batch_size, max_channels) -> x, y [B, C, T], xyz [B, C, 3], valid [B, C]
    #   pick runs with p proportional to n starts (self.runs[:, 2]), one start time per window,
    #   a random subset of <= max_channels rows of that recording, pad the rest and mark them in `valid`.
