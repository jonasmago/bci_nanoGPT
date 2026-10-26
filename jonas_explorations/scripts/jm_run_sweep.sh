#!/usr/bin/env bash
# #jm 2026-10-19: channel encodings x tokenizers -> masked-prediction bits + classification probe.
#   encodings : none (original: wpe only) | learned (channel-name table wce) | rope4d (axial RoPE over t,x,y,z)
#   datasets  : eeg_dac_multichannel (10k_1X1024, V=1024) | eeg_dac_noPost8192 | eeg_dac_corrupt4v3 | eeg_dac_chris512
# Settings = out-eeg-learned (6 layers, 384-d, 32 ch x 64 frames, masked time spans 50 % / 32 frames, 3000 iters),
# except eval_iters 50 every 500 iters (faster) and wandb on. Each job: train (venv_dac) then probe (venv_dac_audio).
# Usage (node 045): nohup bash jm_run_sweep.sh > jm_run_sweep.log 2>&1 &     (4 jobs in parallel: GPUs 4,5,6,6)
set -u
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TRAIN_PY=/data/groups/bci/jonas/venv_dac/bin/python
PROBE_PY=/scratch/jonas/venv_dac_audio/bin/python
ITERS=${ITERS:-3000}

job() {  # index dataset encoding   (GPU by job index: 4 5 6 6, round robin; arrays do not export, hence inline)
  local gpus=(4 5 6 6)
  local i=$1 ds=$2 enc=$3
  local gpu=${gpus[$((i % 4))]} out="out-jm/${2#eeg_dac_}-$3"   # (separate statement: i must be set first)
  if [ -f "$out/probe.done" ]; then echo "[$(date +%H:%M:%S)] skip $out (done)"; return; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] start $out on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $TRAIN_PY train.py config/train_eeg_jm.py --dataset="$ds" --channel_encoding="$enc" \
      --out_dir="$out" --max_iters=$ITERS --lr_decay_iters=$ITERS --wandb_run_name="${out#out-jm/}" > "$out/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] trained $out (exit $?)"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PROBE_PY jonas_explorations/scripts/eeg_probe_jm.py --out_dir "$out" > "$out/probe.log" 2>&1 \
      && touch "$out/probe.done"
  echo "[$(date +%H:%M:%S)] probed $out (exit $?)"
}
export -f job; export TRAIN_PY PROBE_PY ITERS
i=0; jobs_list=()
for ds in eeg_dac_multichannel eeg_dac_noPost8192 eeg_dac_corrupt4v3 eeg_dac_chris512; do
  for enc in none learned rope4d; do jobs_list+=("$i $ds $enc"); i=$((i + 1)); done
done
printf '%s\n' "${jobs_list[@]}" | xargs -P 4 -L 1 bash -c 'job $0 $1 $2'
echo "[$(date +%H:%M:%S)] JM_SWEEP_DONE"
