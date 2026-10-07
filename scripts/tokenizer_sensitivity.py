"""Measure DAC sensitivity to local perturbations in real EEG recordings.

Writes per-frame observations and an averaged temporal sensitivity profile for
continuous pre-quantizer encoder changes and discrete code flips. Fixed baseline
normalization isolates encoder sensitivity; recomputed normalization includes the
nonlocal effect of per-chunk mean/std. Results characterize these perturbations,
not a guaranteed information boundary or an automatically optimal mask span.
"""

import argparse
import csv
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


def normalize(data, mean=None, std=None, target_std=0.3, clip=1.0):
    """Return normalized data and statistics, optionally reusing frozen stats."""
    mean = data.mean(-1, keepdims=True) if mean is None else mean
    std = np.maximum(data.std(-1, keepdims=True), 1e-8) if std is None else std
    return np.clip((data - mean) * (target_std / std), -clip, clip), mean, std


def encoder_geometry(encoder):
    """Maximum field and center for DAC's sequential/residual Conv1d encoder.

    Its residual units preserve sample alignment and have shorter identity
    paths, so the convolution path gives the maximum receptive field.
    """
    import torch
    receptive_field, jump, center = 1, 1, 0.0
    for layer in encoder.modules():
        if isinstance(layer, torch.nn.Conv1d):
            kernel, dilation, stride, padding = (layer.kernel_size[0], layer.dilation[0],
                                                 layer.stride[0], layer.padding[0])
            center += ((kernel - 1) * dilation / 2 - padding) * jump
            receptive_field += (kernel - 1) * dilation * jump
            jump *= stride
    return receptive_field, jump, center


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="DAC weights.pth")
    parser.add_argument("--dac-repo", type=Path, help="Optional path for importing the DAC implementation")
    parser.add_argument("--fif", type=Path, action="append", required=True, help="Repeat for several recordings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=60.0, help="Normalization chunk duration; use pretraining chunk length")
    parser.add_argument("--channels", type=int, default=1, help="First N EEG channels from each recording")
    parser.add_argument("--amplitudes", default="0.01,0.1,1.0", help="Comma-separated noise amplitudes in baseline channel SD units")
    parser.add_argument("--locations", default="0.25,0.5,0.75", help="Comma-separated perturbation centers as fractions of the chunk")
    parser.add_argument("--perturbation-s", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    amplitudes = [float(value) for value in args.amplitudes.split(",")]
    locations = [float(value) for value in args.locations.split(",")]
    if (not amplitudes or any(not np.isfinite(value) or value <= 0 for value in amplitudes)
            or not locations or any(not 0 < value < 1 for value in locations)):
        parser.error("Amplitudes must be finite and positive; locations must lie strictly between zero and one.")
    if (args.start_s < 0 or args.duration_s <= 0 or args.channels < 1
            or not 0 < args.perturbation_s < args.duration_s):
        parser.error("Require start >= 0, duration > perturbation > 0 and channels >= 1.")

    import mne
    import torch
    if args.dac_repo:
        sys.path.insert(0, str(args.dac_repo.resolve()))
    from dac.model import DAC

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kwargs = blob["metadata"]["kwargs"]
    model = DAC(**kwargs)
    model.load_state_dict(blob["state_dict"])
    model.to(args.device).eval().requires_grad_(False)
    sample_rate = int(kwargs.get("sample_rate", model.sample_rate))
    receptive_field, hop, frame_center = encoder_geometry(model.encoder)
    if hop != int(np.prod(kwargs["encoder_rates"])):
        raise ValueError("Encoder temporal geometry differs from checkpoint encoder_rates.")

    def encode(wave):
        tensor = torch.from_numpy(wave.astype(np.float32)).unsqueeze(1).to(args.device)
        tensor = model.preprocess(tensor, sample_rate)
        continuous = model.encoder(tensor)
        _, codes, *_ = model.quantizer(continuous)
        return continuous.cpu().numpy(), codes.cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["recording", "mode", "amplitude_sd", "location_fraction", "offset_frames",
              "distance_s", "encoder_rms_change", "encoder_relative_rms_change", "code_flip_fraction"]
    aggregate = {}
    recordings = []
    with (args.output_dir / "observations.csv").open("w", newline="") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in args.fif:
            raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
            raw.pick("eeg", verbose="error")
            if not np.isclose(raw.info["sfreq"], sample_rate):
                raise ValueError(f"{path}: expected {sample_rate} Hz; got {raw.info['sfreq']}. Resample before this experiment.")
            start = int(round(args.start_s * sample_rate))
            stop = min(raw.n_times, start + int(round(args.duration_s * sample_rate)))
            stop = start + (stop - start) // hop * hop
            if stop - start < 2 * receptive_field:
                raise ValueError(f"{path}: chunk needs at least twice the nominal receptive field "
                                 f"({2 * receptive_field / sample_rate:.3f} seconds).")
            for annotation in raw.annotations:
                onset = float(annotation["onset"]) - raw.first_time
                if "boundary" in annotation["description"].lower() and start / sample_rate < onset < stop / sample_rate:
                    raise ValueError(f"{path}: chosen chunk crosses a recording boundary; change --start-s/--duration-s.")
            wave = raw.get_data(picks=list(range(min(args.channels, len(raw.ch_names)))), start=start, stop=stop).astype(np.float32)
            names = raw.ch_names[:wave.shape[0]]
            raw.close()
            if not np.isfinite(wave).all():
                raise ValueError(f"{path}: non-finite EEG values.")
            baseline, mean, std = normalize(wave)
            baseline_features, baseline_codes = encode(baseline)
            frame_times = (np.arange(baseline_features.shape[-1]) * hop + frame_center) / sample_rate
            baseline_rms = np.sqrt(np.mean(baseline_features.astype(np.float64) ** 2, axis=(0, 1)))
            recordings.append({"path": str(path.resolve()), "start_sample": start, "n_samples": wave.shape[-1],
                               "channels": names, "sample_rate": sample_rate})
            for location in locations:
                width = max(1, int(round(args.perturbation_s * sample_rate)))
                center = int(round(location * (wave.shape[-1] - 1)))
                left, right = center - width // 2, center - width // 2 + width
                if left < 0 or right > wave.shape[-1]:
                    raise ValueError("Perturbation interval leaves the chunk; move --locations away from its edges.")
                noise = rng.standard_normal((wave.shape[0], width)).astype(np.float32)
                nearest_frame = int(np.argmin(abs(frame_times - center / sample_rate)))
                for amplitude in amplitudes:
                    perturbed = wave.copy()
                    perturbed[:, left:right] += amplitude * std * noise
                    for mode in ("fixed_normalization", "recomputed_normalization"):
                        normalized, _, _ = normalize(perturbed, mean, std) if mode == "fixed_normalization" else normalize(perturbed)
                        features, codes = encode(normalized)
                        rms = np.sqrt(np.mean((features.astype(np.float64) - baseline_features) ** 2, axis=(0, 1)))
                        relative = rms / np.maximum(baseline_rms, 1e-12)
                        flips = np.mean(codes != baseline_codes, axis=(0, 1))
                        for frame in range(len(rms)):
                            offset = frame - nearest_frame
                            row = dict(recording=str(path), mode=mode, amplitude_sd=amplitude,
                                       location_fraction=location, offset_frames=offset,
                                       distance_s=float(frame_times[frame] - center / sample_rate),
                                       encoder_rms_change=float(rms[frame]), encoder_relative_rms_change=float(relative[frame]),
                                       code_flip_fraction=float(flips[frame]))
                            writer.writerow(row)
                            key = (mode, amplitude, offset)
                            values = aggregate.setdefault(key, np.zeros(5, dtype=np.float64))
                            values += [1, row["distance_s"], rms[frame], relative[frame], flips[frame]]
            print(f"Profiled {path.name}: {wave.shape[0]} channel(s), {wave.shape[-1] / sample_rate:g} seconds.")

    with (args.output_dir / "profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "amplitude_sd", "offset_frames", "n_observations", "mean_distance_s",
                         "encoder_rms_change", "encoder_relative_rms_change", "code_flip_fraction"])
        for (mode, amplitude, offset), totals in sorted(aggregate.items()):
            writer.writerow([mode, amplitude, offset, int(totals[0]), *(totals[1:] / totals[0])])
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
                "nominal_receptive_field_samples": receptive_field, "nominal_receptive_field_s": receptive_field / sample_rate,
                "hop_samples": hop, "first_frame_center_sample": frame_center, "recordings": recordings,
                "amplitudes_sd": amplitudes, "locations": locations, "perturbation_s": args.perturbation_s,
                "seed": args.seed, "preprocessing": {"target_std": 0.3, "clip": 1.0, "filter": "none; uses supplied FIF"},
                "interpretation": "Sensitivity to sampled perturbations; neither zero code flips nor small feature changes prove absence of signal information. Test a range of mask spans."}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Nominal receptive field: {receptive_field / sample_rate:.3f} s. Wrote observations.csv, profile.csv, metadata.json to {args.output_dir}")


if __name__ == "__main__":
    main()
