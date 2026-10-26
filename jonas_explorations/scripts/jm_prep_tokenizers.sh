#!/usr/bin/env bash
# tokenize the same 47+8 recordings with three more DAC checkpoints (jm, 2026-10-19)
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMPY_MADVISE_HUGEPAGE=0
PY=/scratch/jonas/venv_dac_audio/bin/python
R=/data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs
for spec in "eeg_dac_noPost8192:$R/rand10k_noPost/latest/dac/weights.pth" "eeg_dac_corrupt4v3:$R/rand10k_corrupt4_v3/latest/dac/weights.pth" "eeg_dac_chris512:$R/vocab_crt_cw_21/latest/dac/weights.pth"; do
  name=${spec%%:*}; ckpt=${spec#*:}
  if [ -f data/$name/manifest.json ]; then echo "skip $name (exists)"; continue; fi
  echo "[$(date +%H:%M:%S)] preparing $name"
  $PY data/prepare_eeg_tokens.py --ckpt $ckpt --recordings data/eeg_dac_noPost8192/manifest.json --out_dir data/$name --device cuda:0 2>&1 | grep -v Warning | tail -3
done
echo "[$(date +%H:%M:%S)] PREP_DONE"
