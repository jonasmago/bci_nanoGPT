#!/usr/bin/env bash
# #jm fine-tuning runs: smoke, then pretrained vs random init on the 3k rope4d 1024 model
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMPY_MADVISE_HUGEPAGE=0
PY=/scratch/jonas/venv_dac_audio/bin/python
echo "[$(date +%H:%M:%S)] smoke"; $PY jonas_explorations/scripts/eeg_finetune_jm.py --out_dir out-jm/multichannel-rope4d --steps 3 --tasks ssvep 2>&1 | grep -v Warning | tail -4
rm -f jonas_explorations/results/jm_finetune_results.csv
for init in "" "--random_init"; do
  echo "[$(date +%H:%M:%S)] multichannel-rope4d $init"; $PY jonas_explorations/scripts/eeg_finetune_jm.py --out_dir out-jm/multichannel-rope4d $init 2>&1 | grep -v Warning | grep -E "over|reference|Traceback|Error"
done
echo "[$(date +%H:%M:%S)] multichannel-none"; $PY jonas_explorations/scripts/eeg_finetune_jm.py --out_dir out-jm/multichannel-none 2>&1 | grep -v Warning | grep -E "over|Traceback|Error"
echo "[$(date +%H:%M:%S)] chris512-rope4d"; $PY jonas_explorations/scripts/eeg_finetune_jm.py --out_dir out-jm/chris512-rope4d 2>&1 | grep -v Warning | grep -E "over|Traceback|Error"
echo "[$(date +%H:%M:%S)] FINETUNE_DONE"
