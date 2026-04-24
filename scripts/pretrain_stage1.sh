#!/usr/bin/env bash
set -euo pipefail

# ====== Config ======
PROJECT_ROOT="/home/xuwenqian/dengfei"
# PYTHON_BIN="/path/to/miniconda3/envs/time-llm/bin/python"
DATA_ROOT="/home/xuwenqian/dengfei/data/processed_data_joint_30s_small"

EXP_NAME="imu_rqvae_norm_100ep"
STAGE="rqvae"                  # rqvae | joint
EPOCHS=100
BATCH_SIZE=16
DEVICE="cuda:0"

HIDDEN_DIM=256
IMU_ENCODER_LAYERS=4
RQ_NUM_QUANTIZERS=3
LOG_INTERVAL=20

# normalization
NORMALIZE=True
NORM_MODE="device_channel"     # device_channel | channel | global
NORM_REF_DATA_PATH="data_train.h5"

# ====== Run ======
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

# "${PYTHON_BIN}" ./pretrained-encoder/pretrain.py \
python ./pretrained-encoder/pretrain.py \
  --root_path "${DATA_ROOT}" \
  --train_data_path data_train.h5 \
  --test_data_path data_test.h5 \
  --train_label_path label_train.csv \
  --test_label_path label_test.csv \
  --modality raw \
  --normalize "${NORMALIZE}" \
  --norm_mode "${NORM_MODE}" \
  --norm_ref_data_path "${NORM_REF_DATA_PATH}" \
  --stage "${STAGE}" \
  --epoch "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --imu_encoder_layers "${IMU_ENCODER_LAYERS}" \
  --rq_num_quantizers "${RQ_NUM_QUANTIZERS}" \
  --device "${DEVICE}" \
  --log_interval "${LOG_INTERVAL}" \
  --exp_name "${EXP_NAME}"
