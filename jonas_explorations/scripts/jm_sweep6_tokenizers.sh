#!/usr/bin/env bash
# #jm 2026-10-20: sweep-6 tokenizers through the nanoGPT pipeline (GPU 2): vocab512 (our recipe) and the
# coarse codebook of rvq2x512, each with the default time-span masking (comparable with sec. 13's grid)
# and with random masking (the best recipe), 4D RoPE, 3000 iters, then the probe.
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export CUDA_VISIBLE_DEVICES=2 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
R=/data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs
PREP=/scratch/jonas/venv_dac_audio/bin/python
for spec in "eeg_dac_vocab512:$R/rand10k_vocab512/latest/dac/weights.pth:" "eeg_dac_rvq2x512c0:$R/rand10k_rvq2x512/latest/dac/weights.pth:--codebook 0"; do
  IFS=: read -r name ckpt extra <<< "$spec"
  [ -f data/$name/manifest.json ] && { echo "skip prepare $name"; continue; }
  echo "[$(date +%H:%M:%S)] preparing $name"
  $PREP data/prepare_eeg_tokens.py --ckpt "$ckpt" --recordings data/eeg_dac_noPost8192/manifest.json --out_dir data/$name --device cuda:0 $extra 2>&1 | grep -v Warning | tail -2
done
for ds in eeg_dac_vocab512 eeg_dac_rvq2x512c0; do
  for mask in time random; do
    out="out-jm/${ds#eeg_dac_}-rope4d-$mask"; [ -f "$out/probe.done" ] && { echo "skip $out"; continue; }
    mkdir -p "$out"; echo "[$(date +%H:%M:%S)] train $out"
    /data/groups/bci/jonas/venv_dac/bin/python train.py config/train_eeg_jm.py --dataset=$ds --channel_encoding=rope4d \
        --mask_strategy=$mask --out_dir="$out" --wandb_run_name="${out#out-jm/}" > "$out/train.log" 2>&1
    echo "[$(date +%H:%M:%S)] trained (exit $?)"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PREP jonas_explorations/scripts/eeg_probe_jm.py --out_dir "$out" > "$out/probe.log" 2>&1 && touch "$out/probe.done"
    echo "[$(date +%H:%M:%S)] probed (exit $?)"
  done
done
echo "[$(date +%H:%M:%S)] SWEEP6_TOKENIZERS_DONE"
