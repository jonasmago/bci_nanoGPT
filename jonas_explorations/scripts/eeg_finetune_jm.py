"""Supervised fine-tuning of an EEG nanoGPT checkpoint for classification (#jm, 2026-10-19).

The frozen linear probe (eeg_probe_jm.py) barely beats the token embeddings. This is the other
end: the whole transformer plus a linear head is trained on the task, CROSS-SUBJECT
(GroupKFold over subjects: train on ~80 % of the subjects, test on the rest), on the DAC
eval-suite windows: SSVEP (1800 windows, 10 subjects, 12 classes, 8 ch x 32 frames) and
fist_lr (576 windows, 13 subjects, 2 classes, 64 ch x 32 frames).
  model: pretrained checkpoint (or the same architecture from random init = no pretraining)
  head : linear on the per-channel mean of the final hidden states, flattened [C * n_embd]
  train: AdamW, lr 1e-4 (body) / 1e-3 (head), weight decay 0.01, batch 16, `steps` steps, fp32
  score: accuracy per held-out subject, mean +- std over subjects (as the eval suite)
Reference rows ("raw"): the eval suite's own raw-signal classifiers in the same cross-subject
folds (fine spectrum + logreg for SSVEP, CSP+LDA on 8-30 Hz covariances for fist_lr).

Usage: ~/venv_dac_audio/bin/python eeg_finetune_jm.py --out_dir out-jm/multichannel-rope4d [--random_init] [--steps 300]
Appends to jm_finetune_results.csv
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
import torch.nn as nn
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))   # scripts dir + repo root (moved 2026-10-23)
import eeg_probe_jm as P  # noqa: E402  (tokenisation, channel tables, loaders; sets DAC paths)
from model import GPT, GPTConfig  # noqa: E402

OUT = HERE.parent / "results" / "jm_finetune_results.csv"


class Classifier(nn.Module):
    def __init__(self, gpt, n_channels, n_classes):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(n_channels * gpt.config.n_embd, n_classes)

    def forward(self, idx, ids, xyz):
        b, c, f = idx.shape
        kw = dict(channel_ids=ids[None].expand(b, -1), xyz=xyz[None].expand(b, -1, -1),
                  valid_mask=torch.ones(b, c, f, dtype=torch.bool, device=idx.device))
        h = self.gpt(idx, return_hidden=True, **kw).reshape(b, c, f, -1).mean(2)   # per-channel mean over frames
        return self.head(h.reshape(b, -1))


def run_fold(make_model, codes, y, tr, te, ids, xyz, device, steps, batch, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model().to(device)
    opt = torch.optim.AdamW([{"params": model.gpt.parameters(), "lr": 1e-4},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    base_lr = [g["lr"] for g in opt.param_groups]
    yt = torch.from_numpy(y).to(device)
    model.train()
    for it in range(steps):
        for g, lr in zip(opt.param_groups, base_lr):   # 20-step linear warmup, then constant
            g["lr"] = lr * min(1.0, (it + 1) / 20)
        bi = rng.choice(tr, batch, replace=False)
        logits = model(codes[bi].to(device), ids, xyz)
        loss = nn.functional.cross_entropy(logits, yt[bi])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for a in range(0, len(te), 64):
            preds.append(model(codes[te[a:a + 64]].to(device), ids, xyz).argmax(1).cpu())
    return torch.cat(preds).numpy(), float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--random_init", action="store_true", help="same architecture, no pretraining")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tasks", default="ssvep,fist_lr")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ck = torch.load(Path(args.out_dir) / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    manifest = json.loads((HERE.parents[1] / "data" / cfg["dataset"] / "manifest.json").read_text())
    dac = P.load_dac(manifest["tokenizer"]["checkpoint"], args.device)
    enc = ck["model_args"].get("channel_encoding", "none")
    init = "random" if args.random_init else "pretrained"
    print(f"{args.out_dir}: {cfg['dataset']} (V={manifest['codebook_size']}), {enc}, init {init}, {args.steps} steps", flush=True)
    rows = []
    for task in args.tasks.split(","):
        t0 = time.time()
        d = P.load_task(task)
        ids, xyz, _, _ = P.channel_tables(d["names"], manifest)
        ids, xyz = ids.to(args.device), xyz.to(args.device)
        codes = P.tokenize(dac, d["x"], args.device).long()          # [W, C, F] on CPU
        y, groups = d["y"], d["groups"]
        n_classes, n_ch = len(np.unique(y)), codes.shape[1]

        def make_model():
            gpt = P.load_gpt(args.out_dir, "cpu")[0] if not args.random_init else GPT(GPTConfig(**ck["model_args"]))
            return Classifier(gpt, n_ch, n_classes)

        per_subject = {}
        for k, (tr, te) in enumerate(GroupKFold(min(5, len(np.unique(groups)))).split(codes, y, groups)):
            pred, last_loss = run_fold(make_model, codes, y, tr, te, ids, xyz, args.device, args.steps, args.batch, seed=k)
            for s in np.unique(groups[te]):
                m = groups[te] == s
                per_subject[s] = float((pred[m] == y[te][m]).mean())
            print(f"  {task} fold {k}: test subjects {sorted(np.unique(groups[te]))} acc {100 * (pred == y[te]).mean():.1f} "
                  f"(last train loss {last_loss:.2f})", flush=True)
        acc = np.array(list(per_subject.values()))
        rows.append(dict(out_dir=args.out_dir, dataset=cfg["dataset"], channel_encoding=enc, init=init, steps=args.steps,
                         task=task, acc=100 * acc.mean(), std=100 * acc.std(), n_subjects=len(acc)))
        print(f"  {task}: {100 * acc.mean():.1f} +- {100 * acc.std():.1f} over {len(acc)} held-out subjects ({time.time() - t0:.0f}s)", flush=True)
        # raw-signal reference in the same regime (once)
        if not OUT.exists() or not ((pd.read_csv(OUT).out_dir == "raw") & (pd.read_csv(OUT).task == task)).any():
            C = P.C
            if task == "ssvep":
                X, clf = C.fine_spec(d["x"]), C.logreg
            else:
                meta, kw, names, hop, Tmin, orig, recon, codes_, gains = C.load_task(P.TASKS[task], task)
                mask = ~meta.imagined.to_numpy()
                X, clf = C.mu_beta(np.ascontiguousarray(orig[mask]) * gains[mask]), C.csp_lda
            m, s, _ = C.cv_score(X, y, groups, clf, False, "acc", d["blocks"])
            rows.append(dict(out_dir="raw", dataset="", channel_encoding="", init="raw signal", steps=0, task=task,
                             acc=100 * m, std=100 * s, n_subjects=len(np.unique(groups))))
            print(f"  raw-signal reference ({task}, cross-subject): {100 * m:.1f} +- {100 * s:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
