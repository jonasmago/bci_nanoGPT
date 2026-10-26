#!/usr/bin/env bash
# #jm re-run the probe (now with a cross-subject regime) on every finished sweep checkpoint
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMPY_MADVISE_HUGEPAGE=0
for d in out-jm/*/; do
  [ -f "$d/probe.done" ] || continue
  echo "[$(date +%H:%M:%S)] $d"
  /scratch/jonas/venv_dac_audio/bin/python jonas_explorations/scripts/eeg_probe_jm.py --out_dir "${d%/}" 2>&1 | grep -E "xsub|Traceback|Error"
done
echo "REPROBE_DONE"
