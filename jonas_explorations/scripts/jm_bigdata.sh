#!/usr/bin/env bash
# #jm: does the transformer need more recordings? 400 train / 40 val recordings (1024 tokenizer), 4D RoPE, 15k iters, probe.
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
export CUDA_VISIBLE_DEVICES=6 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
R=/data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs
if [ ! -f data/eeg_dac_1024_big/manifest.json ]; then
  echo "[$(date +%H:%M:%S)] preparing 400+40 recordings"
  /scratch/jonas/venv_dac_audio/bin/python data/prepare_eeg_tokens.py --ckpt $R/10k_1X1024/latest/dac/weights.pth \
      --n_train 400 --n_val 40 --out_dir data/eeg_dac_1024_big --device cuda:0 2>&1 | grep -v Warning | tail -3
fi
OUT=out-jm/big1024-rope4d-15k; mkdir -p $OUT
echo "[$(date +%H:%M:%S)] training $OUT"
/data/groups/bci/jonas/venv_dac/bin/python train.py config/train_eeg_jm.py --dataset=eeg_dac_1024_big --channel_encoding=rope4d \
    --out_dir=$OUT --max_iters=15000 --lr_decay_iters=15000 --eval_interval=1000 --wandb_run_name=big1024-rope4d-15k > $OUT/train.log 2>&1
echo "[$(date +%H:%M:%S)] trained (exit $?)"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/jonas/venv_dac_audio/bin/python jonas_explorations/scripts/eeg_probe_jm.py --out_dir $OUT > $OUT/probe.log 2>&1
echo "[$(date +%H:%M:%S)] BIGDATA_DONE"
