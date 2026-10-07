# EEG token transformer on top of nanoGPT — working plan (2026-10-05)

Goal: test whether a transformer trained on discrete EEG tokens learns a representation
that transfers to new subjects and to electrode layouts it has not seen.

## Components

- Tokenizer: bci_descript-audio-codec, `runs/10k_1X1024/latest/dac/weights.pth`.
  1 codebook x 1024 codes, encoder_rates [2,2,2,4] -> hop 32 -> 8 tokens/s/channel.
  Each channel encoded independently as mono; EEGNormalize per chunk (std 0.3, clip +-1).
  Multi-codebook checkpoints are out of scope until the single-codebook path works.
- Transformer: this nanoGPT fork, extended as below. TFM-Tokenizer is a design reference
  only (masked prediction, channel embedding added to token embedding); no code is ported.

## A. Objective switch in nanoGPT (verified on Shakespeare first)

1. `GPTConfig`: add `bidirectional`. MASK id enters the vocab via `meta.pkl`, like SEP.
2. `CausalSelfAttention.forward`: `is_causal = not bidirectional`; slow path skips the
   triangular `masked_fill` when bidirectional.
3. `GPT.forward`: in training compute logits at every position (loss already supports
   `ignore_index=-1`). For scoring, gather hidden states at scored positions before
   `lm_head` so `[B, C*T, V]` is never materialised. Keep the last-position shortcut
   for causal `generate`.
4. `get_batch`: masked mode builds `x` with MASK at chosen positions and `y` with the
   original token there and -1 elsewhere. Mask rate and 80/10/10 corruption are parameters.
5. Logging: report bits (cross-entropy / ln 2). Causal and masked losses are different
   problems and are not compared to each other.
6. `sample.py`: keep `generate` for causal; add a fill-in demo for masked (mask a few
   characters of a real line, print the model's guesses).

## B. EEG data path

1. `data/eeg_dac/prepare.py` writes per recording: codes `[C, T]` uint16, channel names,
   xyz in head coordinates under ONE fixed global transform (never per-subset
   normalisation), recording boundaries, tokenizer checkpoint path + weight hash.
   Channels without positions: look up by name in MNE standard_1005; else drop.
2. `get_batch`: sample (recording, time window, channel subset); return tokens `[B, C, T]`,
   xyz `[B, C, 3]` and frame index `[T]` (broadcast inside attention), validity mask for
   padding when C varies.
3. Budget: windows of 5-10 s, up to 64 channels -> <= ~5k tokens per sample.

## C. Positional encoding — four arms, same code path, flags differ

   a. temporal RoPE + learned channel-name table (baseline, not a final candidate)
   b. temporal RoPE + absolute Fourier-feature embedding of xyz (REVE-shaped: REVE uses
      Fourier features of (x,y,z,t) added to the input, NOT RoPE)
   c. 4D RoPE over (t, x, y, z); head-dim split per axis configurable (e.g. 32/8/8/8 or
      16 each); spatial axes rescaled with a fixed transform and their own base
   d. 4D RoPE + absolute Fourier xyz
   Coordinates are kept at masked positions. RoPE is relative: a token does not know its
   absolute scalp location from RoPE alone, which is the reason arms b and d exist.

## D. Masking — three arms, span lengths from the sensitivity profile (E)

   1. time spans across all channels (temporal inference)
   2. whole channels / spatial neighbourhoods (cross-channel inference)
   3. mixture

## E. Tokenizer sensitivity profile (before D)

   Perturb several recordings at several amplitudes; measure change in continuous encoder
   output AND fraction of token ids that flip, as a function of temporal distance from the
   perturbation. Two versions: frozen normalisation statistics (encoder sensitivity) and
   full preprocessing (end-to-end; `prepare.py` normalises over each 60 s chunk).
   Nominal receptive field is ~5 s; effective field is unknown. Output: a RANGE of spans
   to test, not a chosen span.

## F. Evaluation

   - Intrinsic: bits per masked token on a FIXED suite of masking challenges (identical
     validation windows and masks for every model, independent of its training mask).
   - Downstream: frozen transformer embeddings -> linear classifier on the labelled
     windows in `/data/groups/bci/datasets/processed/v8_sets/classifier_eval/`
     (reuse parquet layout and scoring code from `bci_descript-audio-codec/scripts/eval`).
     Add subject-disjoint GroupKFold for every task; current TASK_CFG is within-subject
     for all tasks except eyes (correct for codec retention, not for transfer).
     Track two held-out levels separately: unseen by the probe vs unseen by tokenizer and
     transformer pretraining (known leakage: some SSVEP/ERP subjects are in the 10k corpus).
     Held-out montage needs at least one dataset with a different cap.
   - Confirm best combinations across seeds.

## G. Order of work

   1. A on Shakespeare, with the fill-in demo.
   2. B prep on the existing 56-recording sample + E sensitivity profile.
   3. One EEG baseline end to end (arm C.a, masking D.1) + subject-disjoint probe working.
   4. Positional encodings C.a-d under one fixed masking policy.
   5. Masking policies D.1-3 on the promising encodings; seeds.

## H. Later / out of scope now

   Multiple codebooks (delay pattern or summed embeddings); reference/montage handling;
   scaling to the full 10k corpus via Slurm; block-causal multi-channel variant.
