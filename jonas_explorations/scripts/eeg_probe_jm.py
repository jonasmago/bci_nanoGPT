"""Classification probe for EEG nanoGPT checkpoints (#jm, 2026-10-19).

Question: does the transformer's representation of the tokens carry task information, and
how does that depend on the tokenizer (DAC checkpoint) and the channel encoding?

For one checkpoint (out_dir/ckpt.pt, trained on data/<dataset>): the eval-suite windows of
the DAC repo (fist_lr: 576 real windows x 64 ch x 4 s, within-subject; SSVEP: 1800 x 8 ch x
4 s, block-grouped) are tokenised with the SAME DAC checkpoint the model was trained on
(manifest.json), arranged as grids [B, C, 32 frames] with the manifest's channel ids and
standard_1005 xyz (head metres, as prepare.py), and run through the model unmasked. The final
hidden states are projected to 16 dims (label-free PCA fit on these states), pooled per
channel (mean + std over frames) and scored with the eval suite's logistic-regression CV.
Control: the same readout on the token embeddings (wte) alone, i.e. no context.

Runs in the DAC venv (torch 2.0, dac + sklearn): ~/venv_dac_audio/bin/python eeg_probe_jm.py --out_dir out-eeg-xxx
Appends to jm_probe_results.csv (rows: out_dir, dataset, channel_encoding, bits, task, readout, acc, std).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]          # jonas_explorations/scripts -> repo root (moved 2026-10-23)
DAC_REPO = Path("/data/groups/bci/jonas/workspace/bci_descript-audio-codec")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(DAC_REPO))
sys.path.insert(0, str(DAC_REPO / "scripts" / "eval"))
import classify as C  # noqa: E402  (thread caps on import)
from prepare_eeg_tokens import canonical_channel_name, standard_head_positions  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

OUT = HERE.parent / "results" / "jm_probe_results.csv"
TASKS = {  # model-independent input windows cached by the DAC eval suite
    "fist_lr": DAC_REPO / "scripts/eval/results/1k_1X16/1x16_hop32_76e4382e_s9",
    "ssvep":   DAC_REPO / "scripts/eval/results/1k_1X16/ssvep_1x16_hop32_76e4382e",
}
PCA_DIMS = 16


def load_gpt(out_dir, device):
    ck = torch.load(Path(out_dir) / "ckpt.pt", map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["model_args"]))
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}  # torch.compile prefix
    model.load_state_dict(state)
    return model.to(device).eval(), ck


def load_dac(ckpt, device):
    from dac.model import DAC
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = DAC(**blob["metadata"]["kwargs"]); m.load_state_dict(blob["state_dict"])
    return m.to(device).eval()


def load_task(task):
    meta, kw, names, hop, Tmin, orig, recon, codes, gains = C.load_task(TASKS[task], task)
    del recon, codes, gains
    cfg = C.TASK_CFG[task]
    y = np.searchsorted(sorted(meta.label.unique()), meta.label.to_numpy())
    mask = ~meta.imagined.to_numpy() if cfg.get("real_imag") else np.ones(len(meta), bool)
    gcol = cfg.get("group_col")
    return dict(x=np.ascontiguousarray(orig[mask]), names=names, y=y[mask], groups=meta.subject.to_numpy()[mask],
                blocks=meta[gcol].to_numpy()[mask] if gcol else None)


@torch.no_grad()
def tokenize(dac, x, device, bs=512):
    """x [W, C, T] normalised -> codes [W, C, T // hop] (first codebook)."""
    W, Cn, T = x.shape
    flat = x.reshape(W * Cn, T)
    out = []
    for a in range(0, len(flat), bs):
        xb = torch.from_numpy(np.ascontiguousarray(flat[a:a + bs], dtype=np.float32)).to(device)[:, None]
        out.append(dac.encode(xb)[1][:, 0].cpu())
    return torch.cat(out).reshape(W, Cn, -1)


def channel_tables(names, manifest):
    """channel ids (manifest table, 0 = unknown) and xyz in head metres (standard_1005 template)."""
    table = {n: i for i, n in enumerate(manifest["channel_names"])}
    pos = standard_head_positions()
    canon = [canonical_channel_name(n) or n for n in names]
    ids = torch.tensor([table.get(n, 0) for n in canon], dtype=torch.long)
    xyz = torch.tensor(np.stack([pos.get(n, np.zeros(3)) for n in canon]), dtype=torch.float32)
    missing = [n for n in canon if n not in pos]
    unknown = [n for n in canon if n not in table]
    return ids, xyz, missing, unknown


@torch.no_grad()
def hidden_states(model, codes, ids, xyz, device, bs=32):
    """codes [W, C, F] -> (hidden [W, C, F, d], embedding [W, C, F, d])."""
    W, Cn, F = codes.shape
    H, E = [], []
    for a in range(0, W, bs):
        idx = codes[a:a + bs].to(device)
        b = idx.shape[0]
        kw = dict(channel_ids=ids[None].expand(b, -1).to(device), xyz=xyz[None].expand(b, -1, -1).to(device),
                  valid_mask=torch.ones(b, Cn, F, dtype=torch.bool, device=device))
        H.append(model(idx, return_hidden=True, **kw).float().reshape(b, Cn, F, -1).cpu())
        E.append(model.token_vectors(idx).float().cpu())   # wte, or the projected codebook vectors (token_input='codebook')
    return torch.cat(H), torch.cat(E)


def pooled_features(Z, n_pca=PCA_DIMS):
    """Z [W, C, F, d] -> [W, C * 2 * n_pca]: label-free PCA per frame, then mean + std over frames per channel."""
    W, Cn, F, d = Z.shape
    flat = Z.reshape(-1, d)
    sub = flat[torch.randperm(len(flat), generator=torch.Generator().manual_seed(0))[:200_000]]
    mean = sub.mean(0)
    _, _, Vt = torch.linalg.svd(sub - mean, full_matrices=False)
    P = Vt[:n_pca].T
    proj = ((flat - mean) @ P).reshape(W, Cn, F, n_pca)
    return torch.cat([proj.mean(2), proj.std(2)], -1).reshape(W, -1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    model, ck = load_gpt(args.out_dir, args.device)
    cfg = ck["config"]
    manifest = json.loads((HERE.parents[1] / "data" / cfg["dataset"] / "manifest.json").read_text())
    dac = load_dac(manifest["tokenizer"]["checkpoint"], args.device)
    enc = ck["model_args"].get("channel_encoding", "none")
    bits = float(ck["best_val_loss"]) / np.log(2)
    print(f"{args.out_dir}: dataset {cfg['dataset']} (V={manifest['codebook_size']}), channel_encoding {enc}, "
          f"best val {bits:.3f} bits/masked token, iter {ck['iter_num']}", flush=True)
    rows = []
    for task in TASKS:
        d = load_task(task)
        ids, xyz, missing, unknown = channel_tables(d["names"], manifest)
        if missing or unknown:
            print(f"  {task}: {len(missing)} channels without template position, {len(unknown)} not in the channel table", flush=True)
        codes = tokenize(dac, d["x"], args.device)
        H, E = hidden_states(model, codes, ids, xyz, args.device)
        for readout, Z in (("gpt_state", H), ("embedding", E)):
            X = pooled_features(Z)
            # within-subject (the eval suite's regime) and cross-subject (GroupKFold over subjects: the
            # setting a subject-independent decoder needs)
            for regime, within in (("", True), ("_xsub", False)):
                acc, std, _ = C.cv_score(X, d["y"], d["groups"], C.logreg, within, "acc", d["blocks"])
                rows.append(dict(out_dir=args.out_dir, dataset=cfg["dataset"], channel_encoding=enc, bits=bits,
                                 task=task, readout=readout + regime, acc=100 * acc, std=100 * std))
                print(f"  {task:8s} {readout + regime:16s} {100 * acc:5.1f} +- {100 * std:4.1f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
