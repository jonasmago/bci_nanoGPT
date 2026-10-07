"""Frozen transformer embeddings and a subject-disjoint linear EEG probe.

--reencode consumes labeled FIF intervals with the checkpoint's recording-aligned
normalization. Existing codec caches require matching preprocessing declarations
and coordinate fingerprints, or explicit diagnostic mismatch opt-in. Tokenizer
hashes must always match. Probe folds alone cannot establish that subjects were
absent from tokenizer or transformer pretraining.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "4")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tokenizer(dataset_meta, tokenizer_path):
    expected = dataset_meta.get("tokenizer", {}).get("sha256")
    if not expected:
        raise ValueError("Transformer checkpoint has no dataset_meta.tokenizer.sha256; "
                         "tokenizer compatibility cannot be established.")
    actual = sha256(tokenizer_path)
    if actual != expected:
        raise ValueError(f"Tokenizer mismatch: cache {actual}, transformer {expected}. "
                         "Matching vocabulary sizes do not imply matching codes.")
    return actual


def subject_disjoint_probe(features, labels, subjects, n_splits=5, seed=0, metric="balanced"):
    """Fit preprocessing inside each training fold and return OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, subjects = np.asarray(labels).astype(str), np.asarray(subjects).astype(str)
    if len(features) != len(labels) or len(labels) != len(subjects):
        raise ValueError("Features, labels and subjects must have matching row counts.")
    if not np.isfinite(features).all():
        raise ValueError("Probe features must be finite.")
    n_subjects = len(np.unique(subjects))
    if n_subjects < 2 or n_splits < 2:
        raise ValueError("Subject-disjoint evaluation needs at least two subjects and folds.")
    classes = set(labels)
    if len(classes) < 2:
        raise ValueError("Classification needs at least two labels.")
    score = balanced_accuracy_score if metric == "balanced" else accuracy_score
    predictions = np.empty_like(labels)
    fold_ids = np.full(len(labels), -1, dtype=int)
    folds = []
    for fold, (train, test) in enumerate(GroupKFold(min(n_splits, n_subjects)).split(features, labels, subjects)):
        train_subjects, test_subjects = set(subjects[train]), set(subjects[test])
        if train_subjects & test_subjects:
            raise AssertionError("Subject crossed the probe train/test boundary.")
        if set(labels[train]) != classes:
            raise ValueError(f"Fold {fold} training data lacks a class. Use more subjects or fewer folds.")
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, max_iter=3000, random_state=seed))
        classifier.fit(features[train], labels[train])
        predictions[test] = classifier.predict(features[test])
        fold_ids[test] = fold
        folds.append({"fold": fold, "train_subjects": sorted(train_subjects),
                      "test_subjects": sorted(test_subjects), "test_windows": len(test),
                      "score": float(score(labels[test], predictions[test]))})
    per_subject = {}
    for subject in np.unique(subjects):
        selected = subjects == subject
        per_subject[subject] = {"score": float(score(labels[selected], predictions[selected])),
                                "n_windows": int(selected.sum()),
                                "labels": sorted(set(labels[selected]))}
    scores = [item["score"] for item in per_subject.values()]
    report = {"metric": metric, "cv": "subject-disjoint GroupKFold",
              "n_folds": len(folds), "n_subjects": n_subjects, "n_windows": len(labels),
              "classes": sorted(classes), "subject_mean": float(np.mean(scores)),
              "subject_std": float(np.std(scores)), "pooled_score": float(score(labels, predictions)),
              "per_subject": per_subject, "folds": folds,
              "balanced_accuracy_definition": "mean recall over labels present in each scored subset"}
    return report, predictions, fold_ids


def preprocessing_spec(dataset_meta):
    """Explicit effective semantics, including the original v1 prep defaults."""
    p = dict(dataset_meta.get("preprocessing", {}))
    required = ("target_std", "clip", "chunk_s", "min_chunk_s", "normalization")
    if any(key not in p for key in required):
        raise ValueError("Checkpoint lacks the preprocessing declaration needed for re-encoding.")
    if p["normalization"] != "per-channel per-encoding-chunk":
        raise ValueError("Unsupported tokenizer normalization.")
    p.setdefault("chunk_alignment", "recording_start")
    p.setdefault("additional_filter", "none")
    p.setdefault("bad_channel_handling", "as_stored")
    if (p["chunk_alignment"] != "recording_start" or p["additional_filter"] != "none"
            or p["bad_channel_handling"] != "as_stored"):
        raise ValueError("This encoder only supports the nanoGPT recording-aligned, as-stored FIF preparation path.")
    return {key: p[key] for key in (*required, "chunk_alignment", "additional_filter", "bad_channel_handling")}


def cache_preprocessing_check(cache_dir, dataset_meta, allow_mismatch=False):
    expected = preprocessing_spec(dataset_meta)
    declaration = Path(cache_dir) / "probe_preprocessing.json"
    actual = json.loads(declaration.read_text()) if declaration.exists() else None
    mismatches = []
    if actual is None:
        mismatches.append("cache has no probe_preprocessing.json; legacy codec eval used an extra 1–40 Hz filter and per-window normalization")
    else:
        if actual.get("preprocessing") != expected:
            mismatches.append("cache preprocessing differs from the transformer training declaration")
        if actual.get("tokenizer_sha256") != dataset_meta["tokenizer"]["sha256"]:
            mismatches.append("cache preprocessing declaration names another tokenizer")
        if actual.get("coordinate_policy") != dataset_meta.get("coordinates", {}).get("policy", "recording"):
            mismatches.append("cache coordinate policy differs from training")
    if mismatches and not allow_mismatch:
        raise ValueError("; ".join(mismatches) + ". Use --reencode, or explicitly allow diagnostic mismatch with --allow-preprocessing-mismatch.")
    return {"matched": not mismatches, "diagnostic_only": bool(mismatches), "mismatches": mismatches,
            "expected": expected, "cache_declaration": actual}


def select_labels(metadata, args):
    if not {"subject", "label"}.issubset(metadata) or metadata[["subject", "label"]].isna().any().any():
        raise ValueError("Every label row needs a nonmissing subject ID and label.")
    rows = metadata.copy()
    if "task" in rows:
        rows = rows[rows.task == args.task]
    excluded = json.loads(args.exclude_subjects.read_text()) if args.exclude_subjects else []
    if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
        raise ValueError("--exclude-subjects must contain a JSON list of strings.")
    rows = rows[~rows.subject.astype(str).isin(excluded)]
    if args.subjects:
        rows = rows[rows.subject.astype(str).isin(args.subjects.split(","))]
    if args.max_subjects:
        subjects = sorted(rows.subject.astype(str).unique())[:args.max_subjects]
        rows = rows[rows.subject.astype(str).isin(subjects)]
    if args.windows_per_class:
        rows = rows.groupby(["subject", "label"], sort=True, group_keys=False).head(args.windows_per_class)
    if rows.empty:
        raise ValueError("No labeled windows remain after selection.")
    return rows, excluded


def pool_window(model, codes, xyz, names, dataset_meta, args):
    import torch
    codes = np.asarray(codes, dtype=np.int64)
    frames = min(codes.shape[-1], args.max_frames or codes.shape[-1], model.config.block_size // len(names))
    if frames < 1:
        raise ValueError("No complete frame fits the context; reduce --max-channels.")
    if codes.min() < 0 or codes.max() >= dataset_meta["tokenizer"]["codebook_size"]:
        raise ValueError("Codes exceed the checkpoint tokenizer vocabulary.")
    table = {name: index for index, name in enumerate(dataset_meta.get("channel_names", []))}
    if model.config.channel_encoding == "learned" and not table:
        raise ValueError("Checkpoint lacks the learned channel vocabulary.")
    tokens = torch.tensor(codes[:, :frames], device=args.device)[None]
    features = model.pool_features(tokens, xyz=torch.tensor(xyz, dtype=torch.float32, device=args.device)[None],
                                  channel_ids=torch.tensor([[table.get(name, 0) for name in names]], device=args.device),
                                  times=torch.arange(frames, dtype=torch.float32, device=args.device)[None],
                                  valid_mask=torch.ones_like(tokens, dtype=torch.bool))
    return features[0].cpu().numpy(), frames


def cached_features(model, dataset_meta, args):
    import pandas as pd
    from data.eeg_dac_multichannel.prepare import canonical_channel_name, coordinate_fingerprint, standard_head_positions
    provenance = cache_preprocessing_check(args.cache_dir, dataset_meta, args.allow_preprocessing_mismatch)
    snapshot = args.cache_dir / "weights_snapshot.pth"
    tokenizer_path = snapshot if snapshot.exists() else args.tokenizer_checkpoint
    if tokenizer_path is None:
        raise ValueError("Cache needs weights_snapshot.pth or its exact --tokenizer-checkpoint.")
    verify_tokenizer(dataset_meta, tokenizer_path)
    metadata = pd.read_parquet(args.cache_dir / f"{args.task}_windows.parquet")
    metadata["cache_row"] = np.arange(len(metadata))
    rows, excluded = select_labels(metadata, args)
    codes = np.load(args.cache_dir / f"{args.task}_arrays" / "codes.npy", mmap_mode="r")
    names = json.loads((args.cache_dir / f"{args.task}_ch_names.json").read_text())
    if codes.ndim != 4 or codes.shape[2] != 1 or codes.shape[:2] != (len(metadata), len(names)):
        raise ValueError("Cache arrays must be aligned [windows,channels,1,frames].")
    canonical = [canonical_channel_name(name) for name in names]
    coordinate_path = args.cache_dir / f"{args.task}_xyz.npy"
    declaration = provenance["cache_declaration"] or {}
    if coordinate_path.exists():
        full_xyz = np.load(coordinate_path, allow_pickle=False)
        if full_xyz.shape != (len(names), 3) or not np.isfinite(full_xyz).all():
            raise ValueError("Cached coordinates must be finite head-metre [channels,3] values.")
        if coordinate_fingerprint(names, full_xyz) != declaration.get("coordinate_sha256"):
            raise ValueError("Cached coordinate fingerprint differs from the preprocessing declaration.")
        keep = [index for index, name in enumerate(canonical) if name is not None][:args.max_channels]
        coordinate_source = "cache head-metre coordinates with verified fingerprint"
    else:
        message = "cache lacks per-channel coordinates and their fingerprint; template estimates are diagnostic only"
        if not args.allow_preprocessing_mismatch:
            raise ValueError(message + ". Use --reencode.")
        provenance["mismatches"].append(message)
        provenance.update(matched=False, diagnostic_only=True)
        template = standard_head_positions()
        keep = [index for index, name in enumerate(canonical) if name in template][:args.max_channels]
        full_xyz = np.asarray([template.get(name, [np.nan] * 3) for name in canonical])
        coordinate_source = "standard_1005 template (estimated; diagnostic mismatch)"
    if not keep:
        raise ValueError("No cached channels have template coordinates.")
    names = [canonical[index] for index in keep]
    xyz = full_xyz[keep]
    feats, used = [], []
    for row in rows.itertuples():
        feature, frames = pool_window(model, codes[row.cache_row, keep, 0, :], xyz, names, dataset_meta, args)
        feats.append(feature)
        used.append(frames)
    rows = rows.copy()
    rows["frames_used"] = used
    return np.stack(feats), rows, dict(mode="existing_cache", excluded_subjects=excluded,
        preprocessing=provenance, coordinate_source=coordinate_source,
        channels=names, cached_frames=codes.shape[-1], skipped=[])


def reencoded_features(model, dataset_meta, args):
    """Encode recording-aligned normalization chunks, then select labeled frames.

    Synthetic recording boundaries cannot be silently crossed. A label whose
    normalization chunk crosses a BAD boundary is skipped and reported; the
    script never invents a different short-window preprocessing policy.
    """
    import mne
    import pandas as pd
    import torch
    from data.eeg_dac_multichannel.prepare import normalize_segment, resolve_channel_positions, source_description
    if not args.data_root:
        raise ValueError("--reencode requires --data-root containing source FIFs and labels.parquet.")
    spec = preprocessing_spec(dataset_meta)
    tokenizer = dataset_meta["tokenizer"]
    checkpoint_path = args.tokenizer_checkpoint or Path(tokenizer["checkpoint"])
    verify_tokenizer(dataset_meta, checkpoint_path)
    if args.dac_repo:
        sys.path.insert(0, str(args.dac_repo))
    from dac.model import DAC
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kwargs = blob["metadata"]["kwargs"]
    if kwargs["n_codebooks"] != 1 or kwargs["codebook_size"] != tokenizer["codebook_size"]:
        raise ValueError("Tokenizer metadata differs from transformer pretraining.")
    codec = DAC(**kwargs)
    codec.load_state_dict(blob["state_dict"])
    codec.to(args.device).eval().requires_grad_(False)
    del blob
    labels_path = args.labels or args.data_root / "labels.parquet"
    metadata = pd.read_parquet(labels_path)
    metadata["label_row"] = np.arange(len(metadata))
    rows, excluded = select_labels(metadata, args)
    required = {"path", "start_sample", "duration_s"}
    if not required.issubset(rows):
        raise ValueError(f"Labels need {sorted(required)}.")
    sample_rate, hop = int(tokenizer["sample_rate"]), int(tokenizer["hop"])
    chunk_samples = int(float(spec["chunk_s"]) * sample_rate) // hop * hop
    minimum = int(float(spec["min_chunk_s"]) * sample_rate)
    if chunk_samples < max(hop, minimum):
        raise ValueError("Invalid checkpoint normalization chunk duration.")
    features, accepted, skipped, recordings = [], [], [], []
    policy = dataset_meta.get("coordinates", {}).get("policy", "recording")
    source_max_seconds = dataset_meta.get("args", {}).get("max_seconds")
    for relative, group in rows.sort_values(["path", "start_sample"]).groupby("path", sort=False):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Label source path leaves --data-root.")
        # classifier_eval deliberately symlinks some recordings from v8/10k.
        path = (args.data_root / relative_path).resolve()
        raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
        raw.pick("eeg", verbose="error")
        available_samples = min(raw.n_times, int(float(source_max_seconds) * sample_rate)) if source_max_seconds else raw.n_times
        if not np.isclose(raw.info["sfreq"], sample_rate):
            raise ValueError(f"{path}: sample rate differs from tokenizer; no implicit resampling is performed.")
        picks, names, xyz, sources, dropped = resolve_channel_positions(raw, policy)
        picks, names, xyz, sources = picks[:args.max_channels], names[:args.max_channels], xyz[:args.max_channels], sources[:args.max_channels]
        if not picks:
            raise ValueError(f"{path}: no channels with usable coordinates.")
        boundaries = [int(round((annotation["onset"] - raw.first_time) * sample_rate))
                      for annotation in raw.annotations if "boundary" in annotation["description"].lower()]
        description = source_description(raw)
        recordings.append(dict(path=str(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                               upstream_preprocessing=description, channels=names, coordinate_sources=sources,
                               dropped_channels=dropped))
        previous_chunk, encoded = None, None
        for row in group.to_dict("records"):
            onset = int(row["start_sample"])
            stop = onset + int(round(float(row["duration_s"]) * sample_rate))
            chunk_start = onset // chunk_samples * chunk_samples
            chunk_stop = min(chunk_start + chunk_samples, available_samples)
            chunk_stop = chunk_start + (chunk_stop - chunk_start) // hop * hop
            reason = None
            if onset < 0 or stop > available_samples or stop <= onset:
                reason = "label interval leaves the checkpoint's recording crop or is empty"
            elif stop > chunk_stop:
                reason = "label crosses an independent tokenizer chunk"
            elif chunk_stop - chunk_start < minimum:
                reason = "normalization chunk shorter than checkpoint min_chunk_s"
            elif any(chunk_start < boundary < chunk_stop for boundary in boundaries):
                reason = "normalization chunk crosses a synthetic recording boundary"
            first, last = (onset - chunk_start + hop - 1) // hop, (stop - chunk_start) // hop
            if not reason and last <= first:
                reason = "label contains no complete token frame"
            if reason:
                skipped.append(dict(label_row=int(row["label_row"]), reason=reason, path=relative))
                continue
            if previous_chunk != chunk_start:
                wave = raw.get_data(picks=picks, start=chunk_start, stop=chunk_stop).astype(np.float32)
                if not np.isfinite(wave).all():
                    raise ValueError(f"{path}: nonfinite source EEG.")
                normalized = normalize_segment(wave, target_std=float(spec["target_std"]), clip=float(spec["clip"]))
                tensor = torch.from_numpy(normalized).unsqueeze(1).to(args.device)
                _, codes, *_ = codec.encode(codec.preprocess(tensor, sample_rate))
                encoded = codes[:, 0].cpu().numpy()
                if encoded.shape != (len(picks), (chunk_stop - chunk_start) // hop):
                    raise ValueError("Unexpected tokenizer temporal alignment.")
                previous_chunk = chunk_start
            feature, frames = pool_window(model, encoded[:, first:last], xyz, names, dataset_meta, args)
            features.append(feature)
            row.update(frames_used=frames, complete_label_frames=last-first,
                       encoding_chunk_start_sample=chunk_start, encoding_chunk_stop_sample=chunk_stop,
                       scored_start_sample=chunk_start+first*hop, scored_stop_sample=chunk_start+(first+frames)*hop)
            accepted.append(row)
        raw.close()
    if not accepted:
        raise ValueError("No labels can be encoded with the checkpoint preprocessing: " + json.dumps(skipped[:5]))
    return np.stack(features), pd.DataFrame(accepted), dict(mode="reencode", excluded_subjects=excluded,
        preprocessing={"matched": True, "diagnostic_only": False, "effective": spec,
                       "note": "No additional filter/interpolation; existing FIF upstream preprocessing is retained and recorded."},
        coordinate_policy=policy, source_max_seconds=source_max_seconds, source_recordings=recordings, labels_path=str(labels_path.resolve()),
        labels_sha256=sha256(labels_path), skipped=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cache-dir", type=Path)
    inputs.add_argument("--reencode", action="store_true", help="Use recording-aligned normalization from the transformer checkpoint")
    parser.add_argument("--data-root", type=Path, help="classifier_eval/<dataset> containing source FIFs")
    parser.add_argument("--labels", type=Path, help="Defaults to data-root/labels.parquet")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path)
    parser.add_argument("--dac-repo", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for CLI compatibility; features are extracted one window at a time")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", choices=("balanced", "accuracy"), default="balanced")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-channels", type=int)
    parser.add_argument("--subjects", help="Comma-separated exact IDs for a bounded probe")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--windows-per-class", type=int, help="First N windows per subject/label; bounded smoke tests only")
    parser.add_argument("--exclude-subjects", type=Path)
    parser.add_argument("--allow-preprocessing-mismatch", action="store_true", help="Diagnostic existing-cache runs only; report marks mismatch")
    args = parser.parse_args()
    for name in ("batch_size", "max_frames", "max_channels", "max_subjects", "windows_per_class"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    import torch
    from model import GPT, GPTConfig
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset_meta = checkpoint.get("dataset_meta", {})
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()}
    model.load_state_dict(state)
    model.to(args.device).eval().requires_grad_(False)
    with torch.inference_mode():
        features, metadata, provenance = (reencoded_features if args.reencode else cached_features)(model, dataset_meta, args)
    report, predictions, folds = subject_disjoint_probe(features, metadata.label, metadata.subject, args.folds, args.seed, args.metric)
    report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  task=args.task, tokenizer_sha256=dataset_meta["tokenizer"]["sha256"],
                  provenance=provenance, pooling="valid mean over channels and time",
                  pretraining_subject_exclusion="not established; subjects are excluded only from probe classifier fitting",
                  bounded_smoke=bool(args.windows_per_class or args.max_subjects), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    metadata = metadata.copy()
    metadata["prediction"], metadata["probe_fold"] = predictions, folds
    metadata.to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    print(f"Subject mean {args.metric}: {report['subject_mean']:.4f}; {len(metadata)} windows, "
          f"{len(provenance['skipped'])} skipped; report {args.output}")


if __name__ == "__main__":
    main()
