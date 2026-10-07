"""Tokenize EEG recordings with the EEG-DAC codec into synchronized channel x time grids.

One .npz per recording: codes[C,T] (uint16), xyz[C,3] (head metres), times[T] (frame
indices), segment_ids[T] (independently normalized/encoded 60 s chunks), channel_ids[C],
channel_names[C]. Plus manifest.json (vocab, tokenizer hash, channel-name table, splits).
Only one-codebook DAC checkpoints are supported.

  python data/eeg_dac_multichannel/prepare.py --device cuda:0                 # sample 48/8 recordings from v8
  python data/eeg_dac_multichannel/prepare.py --recordings data/eeg_dac_multichannel/manifest.json ...  # same recordings again
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np

DAC_REPO = "/data/groups/bci/jonas/workspace/bci_descript-audio-codec"
V8 = "/data/groups/bci/datasets/processed/v8_sets/10k"
SR = 256
TARGET_STD, CLIP = 0.3, 1.0
# v8 FIF files carry template positions; TUH linked-ear files have 4 wrong ones (C3/C4/A1/A2),
# so for verified v8 TUH recordings the standard_1005 template position is used instead.
COORDINATE_POLICY = "verified_v8"


@lru_cache(maxsize=1)
def standard_head_positions():
    """Return standard_1005 coordinates transformed by MNE into head metres.

    make_standard_montage itself returns MRI-frame positions; attaching it to
    Info performs the fiducial transform, matching FIF head coordinates.
    """
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(montage.ch_names, SR, ch_types="eeg")
    info.set_montage(montage)
    return {ch["ch_name"]: ch["loc"][:3].copy() for ch in info["chs"]}


def canonical_channel_name(name):
    """Canonical electrode name; reject bipolar derivations and empty names.

    Unfamiliar names remain usable with finite head-frame coordinates. Only template
    fallback requires a known standard electrode name.
    """
    name = re.sub(r"^EEG\s+", "", str(name).strip(), flags=re.I)
    name = re.sub(r"[-_](REF|LE|AVG|AR|A1A2)$", "", name, flags=re.I).strip()
    if not name or any(separator in name for separator in ("-", "/")):
        return None
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    name = aliases.get(name.upper(), name)
    canonical = {key.upper(): key for key in standard_head_positions()}
    return canonical.get(name.upper(), name)


def source_description(raw):
    """Read descriptive provenance only; never treat metadata as executable input."""
    try:
        result = json.loads(raw.info.get("description") or "{}")
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def coordinate_metadata(policy):
    import mne
    template = standard_head_positions()
    packed = json.dumps({name: np.asarray(template[name]).tolist() for name in sorted(template)}, sort_keys=True).encode()
    return dict(frame="head", units="m", fallback_montage="standard_1005", transform="identity",
                policy=policy, resolver_version="2", template_mne_version=mne.__version__,
                template_sha256=hashlib.sha256(packed).hexdigest(),
                correction_rule="verified_v8 TUH canonical standard sites; preserve unverified/digitized headers")


def coordinate_fingerprint(names, xyz):
    digest = hashlib.sha256(json.dumps([str(name) for name in names]).encode())
    digest.update(np.asarray(xyz, dtype="<f4").tobytes())
    return digest.hexdigest()


def resolve_channel_positions(raw, coordinate_policy="recording", return_details=False):
    """Resolve EEG channel head coordinates, marking template fallbacks.

    Return (picks, canonical_names, xyz, coordinate_sources, dropped_names).
    Bipolar channels are excluded even if upstream processing assigned a
    point to them: a single XYZ does not describe a two-electrode derivation.
    """
    from mne.io.constants import FIFF
    from mne.transforms import apply_trans
    if coordinate_policy not in ("recording", "template", "verified_v8"):
        raise ValueError("coordinate_policy must be recording, template, or verified_v8")
    template = standard_head_positions()
    description = source_description(raw)
    verified_v8 = (str(description.get("pipeline", "")).startswith("v8.")
                   and isinstance(description.get("mapping_info"), dict))
    verified_tuh = verified_v8 and "/tuh/" in str(description.get("source_key", ""))
    picks, names, positions, sources, dropped, details = [], [], [], [], [], []
    for i, ch in enumerate(raw.info["chs"]):
        name = canonical_channel_name(ch["ch_name"])
        if name is None:
            dropped.append(ch["ch_name"])
            continue
        xyz = np.asarray(ch["loc"][:3], dtype=float)
        original = xyz.copy()
        frame = ch["coord_frame"]
        source = "v8_template_header" if verified_v8 else "recording_head_unverified"
        if frame == FIFF.FIFFV_COORD_DEVICE and raw.info.get("dev_head_t") is not None:
            xyz = apply_trans(raw.info["dev_head_t"], xyz)
            frame, source = FIFF.FIFFV_COORD_HEAD, "recording_device_to_head_unverified"
        usable = frame == FIFF.FIFFV_COORD_HEAD and np.isfinite(xyz).all() and 1e-6 < np.linalg.norm(xyz) < 0.5
        reason = None
        use_template = coordinate_policy == "template" or (coordinate_policy == "verified_v8" and verified_tuh)
        if use_template and name in template:
            xyz = template[name]
            source = "standard_1005_head_verified_v8_tuh" if verified_tuh and coordinate_policy == "verified_v8" else "standard_1005_head_explicit"
            reason = "v8 TUH template-assigned coordinates: prevent BioSemi-name collision" if verified_tuh else "explicit template policy"
        elif not usable:
            if name not in template:
                dropped.append(ch["ch_name"])
                continue
            xyz, source = template[name], "standard_1005_head_fallback"
            reason = "missing or unusable head coordinate"
        if name in names:
            # An alias collision does not silently become two copies of an electrode.
            dropped.append(ch["ch_name"])
            continue
        picks.append(i)
        names.append(name)
        positions.append(xyz)
        sources.append(source)
        details.append(dict(name=name, original_name=ch["ch_name"], original_xyz_m=[float(v) if np.isfinite(v) else None for v in original],
                            resolved_xyz_m=np.asarray(xyz).tolist(), source=source, reason=reason,
                            changed=not np.allclose(original, xyz, atol=1e-7, rtol=0, equal_nan=True)))
    result = (picks, names, np.asarray(positions, dtype=np.float32).reshape(-1, 3), sources, dropped)
    return (*result, details) if return_details else result


def normalize_segment(x, target_std=TARGET_STD, clip=CLIP):
    """Match training EEGNormalize, with statistics per channel and chunk."""
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True).clip(min=1e-8)
    return np.clip((x - mean) * (target_std / std), -clip, clip)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_recordings(args, split, count, previous_manifest=None):
    """Recordings to encode: those of an earlier manifest, or a random sample of the v8 set."""
    if previous_manifest is not None:
        return [dict(source_path=r["source_path"], dataset=str(r["dataset"]), subject=str(r["subject"]))
                for r in previous_manifest["splits"][split][:count]]
    import pandas as pd
    df = pd.read_parquet(Path(args.data_root) / "manifests" / f"10k_{split}.parquet")
    df = df[~df.is_epoched.astype(bool)].sample(frac=1, random_state=args.seed)  # continuous recordings only
    files = []
    for row in df.itertuples():
        path = Path(args.data_root) / split / row.rel_path
        if path.exists():
            files.append(dict(source_path=str(path), dataset=str(row.dataset), subject=str(row.subject)))
        if len(files) >= count:
            break
    return files


def encode_recording(path, model, args, hop, codebook_size):
    import mne
    import torch
    raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
    raw.pick("eeg", verbose="error")
    if raw.info["sfreq"] != SR:
        raise ValueError(f"expected {SR} Hz, got {raw.info['sfreq']}")
    picks, names, xyz, sources, dropped, coordinate_details = resolve_channel_positions(
        raw, COORDINATE_POLICY, return_details=True)
    picks, names, xyz, sources = picks[:args.max_channels], names[:args.max_channels], xyz[:args.max_channels], sources[:args.max_channels]
    coordinate_details = coordinate_details[:args.max_channels]
    if not picks:
        raise ValueError("no channels with supported electrode positions")
    stop = min(raw.n_times, int(args.max_seconds * SR))
    x = raw.get_data(picks=picks, start=0, stop=stop).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("recording contains nonfinite EEG values")
    chunk_samples = int(args.chunk_s * SR) // hop * hop
    codes, times, segment_ids, segment_bounds = [], [], [], []
    offset = 0
    with torch.no_grad():
        for start in range(0, x.shape[1], chunk_samples):
            seg = x[:, start:start + chunk_samples]
            n = seg.shape[-1] // hop * hop
            if n < int(args.min_chunk_s * SR):
                break
            seg = normalize_segment(seg[:, :n])
            xt = torch.from_numpy(seg).unsqueeze(1).to(args.device)
            _, encoded, *_ = model.encode(model.preprocess(xt, SR))
            array = encoded[:, 0].cpu().numpy()
            if array.shape != (len(picks), n // hop):
                raise ValueError(f"unexpected DAC token shape {array.shape}; expected {(len(picks), n // hop)}")
            if array.min() < 0 or array.max() >= codebook_size:
                raise ValueError("DAC emitted codes outside checkpoint vocabulary")
            count = array.shape[1]
            codes.append(array.astype(np.uint16))
            times.append(start // hop + np.arange(count, dtype=np.int64))
            segment_ids.append(np.full(count, len(codes) - 1, dtype=np.int32))
            segment_bounds.append([offset, offset + count])
            offset += count
    raw.close()
    if not codes:
        raise ValueError("recording has no sufficiently long tokenizer chunks")
    return dict(codes=np.concatenate(codes, axis=1), xyz=xyz, times=np.concatenate(times),
                segment_ids=np.concatenate(segment_ids), channel_names=np.asarray(names)), dict(
                n_ch=len(picks), n_samples=x.shape[1], n_frames=offset,
                segment_bounds=segment_bounds, coordinate_sources=sources, coordinate_details=coordinate_details,
                coordinate_sha256=coordinate_fingerprint(names, xyz), dropped_channels=dropped,
                source_preprocessing=source_description(raw))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=f"{DAC_REPO}/runs/10k_1X1024/latest/dac/weights.pth")
    parser.add_argument("--dac_repo", default=DAC_REPO)
    parser.add_argument("--data_root", default=V8)
    parser.add_argument("--recordings", help="manifest.json of an earlier run: encode exactly the same recordings")
    parser.add_argument("--n_train", type=int, default=48)
    parser.add_argument("--n_val", type=int, default=8)
    parser.add_argument("--max_seconds", type=float, default=600)
    parser.add_argument("--max_channels", type=int, default=64)
    parser.add_argument("--chunk_s", type=float, default=60)
    parser.add_argument("--min_chunk_s", type=float, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not all(np.isfinite(value) for value in (args.max_seconds, args.chunk_s, args.min_chunk_s)):
        parser.error("durations must be finite")
    if args.max_seconds < args.min_chunk_s:
        parser.error("max_seconds must accommodate min_chunk_s")
    if min(args.n_train, args.n_val, args.max_seconds, args.max_channels, args.chunk_s, args.min_chunk_s) <= 0:
        parser.error("recording counts, durations, and channel limits must be positive")
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    args.out_dir = str(out_dir)
    if not args.overwrite and any((out_dir / name).exists() for name in ("manifest.json", "train", "val")):
        parser.error(f"output already exists in {out_dir}; use a new --out_dir or explicit --overwrite")
    previous_manifest = json.loads(Path(args.recordings).read_text()) if args.recordings else None
    sys.path.insert(0, args.dac_repo)
    from dac.model import DAC
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    kw = blob["metadata"]["kwargs"]
    K, V, hop = kw["n_codebooks"], kw["codebook_size"], int(np.prod(kw["encoder_rates"]))
    if K != 1:
        parser.error("only single-codebook checkpoints are supported")
    if V > np.iinfo(np.uint16).max:
        parser.error("codebook and special token must fit uint16")
    if kw.get("sample_rate", SR) != SR:
        parser.error(f"expected a {SR} Hz tokenizer")
    if int(args.chunk_s * SR) // hop * hop < max(hop, int(args.min_chunk_s * SR)):
        parser.error("chunk_s must accommodate min_chunk_s and at least one token")
    model = DAC(**kw)
    model.load_state_dict(blob["state_dict"])
    model.to(args.device).eval()
    del blob
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(format="eeg_multichannel_v1", codebook_size=V, vocab_size=V + 1,
        mask_token_id=V, channel_names=["<UNK>"], tokens_per_s=SR / hop,
        tokenizer=dict(checkpoint=str(Path(args.ckpt).resolve()), sha256=file_sha256(args.ckpt),
                       n_codebooks=K, codebook_size=V, hop=hop, sample_rate=SR),
        preprocessing=dict(target_std=TARGET_STD, clip=CLIP, chunk_s=args.chunk_s,
            min_chunk_s=args.min_chunk_s, normalization="per-channel per-encoding-chunk",
            chunk_alignment="recording_start", additional_filter="none", bad_channel_handling="as_stored",
            source_signal="existing FIF samples; upstream preprocessing retained"),
        coordinates=coordinate_metadata(COORDINATE_POLICY),
        splits={"train": [], "val": []}, args=vars(args))
    name_to_id = {"<UNK>": 0}
    for split, count in (("train", args.n_train), ("val", args.n_val)):
        files, started = pick_recordings(args, split, count, previous_manifest), time.time()
        for i, source in enumerate(files):
            try:
                arrays, details = encode_recording(source["source_path"], model, args, hop, V)
            except Exception as error:
                print(f"skip {source['source_path']}: {error}", flush=True)
                continue
            for name in arrays["channel_names"]:  # channel-name vocabulary is built from TRAIN recordings only
                if split == "train" and name not in name_to_id:
                    name_to_id[str(name)] = len(name_to_id)
            arrays["channel_ids"] = np.asarray([name_to_id.get(str(name), 0) for name in arrays["channel_names"]], dtype=np.int64)
            relative = f"{split}/{i:05d}.npz"
            (out_dir / split).mkdir(exist_ok=True)
            np.savez(out_dir / relative, **arrays)
            record = dict(source, **details, path=relative)
            manifest["splits"][split].append(record)
            print(f"[{split} {i+1}/{len(files)}] {source['dataset']}/{source['subject']}: "
                  f"{details['n_ch']} ch x {details['n_frames']} frames ({time.time()-started:.0f}s)", flush=True)
        if not manifest["splits"][split]:
            raise RuntimeError(f"no usable {split} recordings; no valid dataset manifest was written")
    manifest["channel_names"] = list(name_to_id)
    manifest["coordinates"]["dataset_sha256"] = hashlib.sha256(json.dumps([
        [record["path"], record["coordinate_sha256"]] for records in manifest["splits"].values() for record in records
    ], sort_keys=True).encode()).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
