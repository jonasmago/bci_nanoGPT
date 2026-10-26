#!/usr/bin/env bash
# #jm masking comparison for EEG pretraining (1024 tokenizer, 4D RoPE, 3000 iters): strategy x ratio x span.
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
job() {  # index strategy ratio span
  local gpus=(4 5 4 5); local i=$1 st=$2 ratio=$3 span=$4
  local gpu=${gpus[$((i % 4))]} out="out-jm/mask-$2-r$3-s$4"
  [ -f "$out/probe.done" ] && { echo "skip $out"; return; }
  mkdir -p "$out"; echo "[$(date +%H:%M:%S)] start $out on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu /data/groups/bci/jonas/venv_dac/bin/python train.py config/train_eeg_jm.py --dataset=eeg_dac_multichannel \
      --channel_encoding=rope4d --mask_strategy=$st --mask_ratio=$ratio --mask_span=$span --out_dir="$out" \
      --wandb_run_name="${out#out-jm/}" > "$out/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] trained $out (exit $?)"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/jonas/venv_dac_audio/bin/python jonas_explorations/scripts/eeg_probe_jm.py --out_dir "$out" > "$out/probe.log" 2>&1 && touch "$out/probe.done"
  echo "[$(date +%H:%M:%S)] probed $out (exit $?)"
}
export -f job
printf "%s\n" "0 random 0.5 32" "1 time 0.5 8" "2 channel 0.5 32" "3 mixed 0.5 32" "4 time 0.25 32" "5 time 0.75 32" | xargs -P 4 -L 1 bash -c "job \$0 \$1 \$2 \$3"
echo "[$(date +%H:%M:%S)] MASK_SWEEP_DONE"
