#!/usr/bin/env bash

# ===== model zoo =====
MODEL=mgvq-f16c32; CKPT="/path/to/mgvq-f16c32.pt"; DS_RATE=16
# MODEL=mgvq-f8c32;  CKPT="/path/to/mgvq-f8c32.pt"; DS_RATE=8
# MODEL=mgvq-f32c32; CKPT="/path/to/mgvq-f32c32.pt"; DS_RATE=32

# ===== knobs =====
CODEBOOK_SIZE=32768      # 16384 | 32768
CODEBOOK_GROUPS=4        # 4 | 8
GROUPS_TO_USE=4          # <= CODEBOOK_GROUPS
DATASET=imagenet256p     # imagenet256p | UHDBench2k
DATASET_ROOT="/path/to/dataset" # .../origin/val/ | .../UHDBench/

OUT="./eval_imgs"

# for imagenet 256p reconstruction evaluation
python eval_recon.py \
--vq-model "$MODEL" \
--vq-ckpt "$CKPT" \
--codebook-size "$CODEBOOK_SIZE" \
--codebook-groups "$CODEBOOK_GROUPS" \
--groups-to-use "$GROUPS_TO_USE" \
--eval-dataset "$DATASET" \
--ds-rate "$DS_RATE" \
--path-to-save "$OUT" \
--dataset-root "$DATASET_ROOT" \
--eval-fid

