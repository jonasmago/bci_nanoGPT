> **2026-10-23, vector input:** `token_input='codebook'` (model.py `GPTConfig`, train.py knob, on by default in `config/train_eeg_jm.py`): a token enters as its DAC codebook vector (`<dataset>/codebook.npy`, written by `data/prepare_eeg_tokens.py`, or `--codebook_only` for an existing dataset) through `in_proj = Linear(codebook_dim, n_embd)`; the MASK id gets a learned vector `mask_emb`; `wte` and the weight tying disappear, `lm_head` is a free Linear. `token_input='embedding'` is the original id path. `GPT.token_vectors(idx)` returns the input vectors in both modes (the probe's embeddings-only control uses it).

> **2026-10-23, data layout:** `data/prepare_eeg_tokens.py` (was `data/eeg_dac_multichannel/prepare.py`) tokenises recordings with any DAC checkpoint (`--ckpt`, default = the Chris-parameter 20k baseline) into `data/eeg_dac_<run>/`, reusing the recordings of `data/eeg_dac_noPost8192`. Only that dataset is kept; the manifests of the deleted ones are in `jonas_explorations/data_manifests/` for exact regeneration.

> **2026-10-23:** everything of mine now lives under `jonas_explorations/` — `scripts/` (probe, fine-tune, plot, launchers), `results/` (csv), `logs/`, `jm_originals/` (your files before my edits). Still at the repo root: `model.py`, `train.py`, `configurator.py`, `position.py` (the `#jm`-marked code), `config/train_eeg_jm.py`, `data/eeg_dac_multichannel/prepare.py`. Launchers `cd` to the repo root and call `jonas_explorations/scripts/...`.

# Edits by the "DAC classifier" Claude session (jm), 2026-10-19

Copies of every file as it was before these edits are in `jm_originals/` (same relative
paths); every edited line is marked `#jm`. `diff -u jm_originals/model.py model.py` etc.

| file | change |
|---|---|
| `position.py` | **new**: ported verbatim from branch `eeg-reference` (reviewed implementation). Only `RotaryPositions` is used on master. |
| `model.py` | `channel_encoding='rope4d'`: axial RoPE over (frame, x, y, z) in every attention layer (`RotaryPositions`, dims `rope_dims`, default `32,8,8,8` of the 64-dim head; time base 10000, space base 100 on centimetres — the eeg-reference constants). It **replaces** `wpe` (no position tables at all, time is relative too), the eeg-reference `rope` design. `GPT.forward` takes `xyz [b,c,3]` and `return_hidden=True` (final hidden states, for the probe). `Block`/`CausalSelfAttention` pass the rotary tables through. `get_num_params` / `crop_block_size` guard the missing `wpe`. |
| `train.py` | passes `xyz` from the dataset to the model; `rope_dims` config knob; `rope_dims` in `model_args`. |
| `configurator.py` | string knobs stay strings (`--rope_dims=32,8,8,8` was parsed as a tuple and rejected). |
| `config/train_eeg_jm.py` | **new**: the out-eeg-learned settings as a config file. |
| `eeg_probe_jm.py` | **new**: classification probe (fist_lr / SSVEP windows of the DAC eval suite, tokenised with the checkpoint's own tokenizer, hidden states → PCA-16 → mean+std per channel → eval-suite logistic-regression CV; control: token embeddings only). |
| `jm_run_sweep.sh`, `jm_prep_tokenizers.sh` | **new**: the 3 encodings × 4 tokenizers sweep, and the tokenisation of the same 47+8 recordings with three more DAC checkpoints (`data/eeg_dac_noPost8192`, `eeg_dac_corrupt4v3`, `eeg_dac_chris512`). |

| `data/eeg_dac_multichannel/prepare.py` (2026-10-20) | `--codebook <i>`: keep one residual stream of a multi-codebook (RVQ) checkpoint; the dataset then looks single-codebook (`n_codebooks` 1, `codebook_index` recorded in the manifest). Without the flag, behaviour is unchanged. |

Not touched: `eeg_data.py`, `fill_mask.py`, `sample.py`.
Results: `jm_probe_results.csv`, `out-jm/<tokenizer>-<encoding>/`, wandb project `nanogpt-eeg-jm`,
and section 13 of `bci_descript-audio-codec/notes_jm/lab_notebook_jm.md`.
