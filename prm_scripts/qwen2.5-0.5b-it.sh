#!/bin/bash
# Qwen2.5-0.5B-Instruct Training Script (Single Node, 8 GPUs)

set -e

# Unset AMD ROCm env var early — before Ray starts, so workers don't inherit it
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
export WANDB_ENTITY=Tsinghua-IIIS-AI-Team

# ============ Configuration ============
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
MODEL_NAME=Qwen2.5-0.5B-Instruct

TRAIN_DATA=/home/azanette/Math-PRM/data/gsm8k/train.parquet
VAL_DATA=/home/azanette/Math-PRM/data/gsm8k/test.parquet

CHECKPOINT_DIR=/home/azanette/Math-PRM/checkpoints

# Training hyperparameters
ADVANTAGE_ESTIMATOR=grpo

# Uncomment the following line if you want to run GRPO or RLOO
# ADVANTAGE_ESTIMATOR=grpo
# ADVANTAGE_ESTIMATOR=rloo

LR=1e-6
N_ROLLOUTS=16
N_VAL=32
BATCH_SIZE=256
MINI_BATCH_SIZE=256
MICRO_BATCH_SIZE=8
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
TOTAL_EPOCHS=10

PROJECT_NAME=Qwen25_0.5B_IT_PRM_Debug

EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}

# ============ Ray Setup (Single Node) ============
RAY_TMPDIR=/tmp/ray_${USER}_$$

# Aggressively clean up any stale Ray state
ray stop --force 2>/dev/null || true
# Kill any remaining Ray processes (GCS, raylet, etc.) from old sessions
pkill -9 -u $(whoami) -f "ray::" 2>/dev/null || true
pkill -9 -u $(whoami) -f "gcs_server" 2>/dev/null || true
pkill -9 -u $(whoami) -f "raylet" 2>/dev/null || true
sleep 2
# Clean stale Ray temp dirs
rm -rf /tmp/ray/session_* 2>/dev/null || true
rm -rf ${RAY_TMPDIR} 2>/dev/null || true

# Start Ray and capture its output to extract the address
RAY_START_OUTPUT=$(ray start --head --num-gpus 8 --temp-dir=${RAY_TMPDIR} --port=0 --dashboard-port=0 2>&1)
echo "$RAY_START_OUTPUT"

# Parse the address from "ray start --address='<ip>:<port>'" in the output
export RAY_ADDRESS=$(echo "$RAY_START_OUTPUT" | grep -oP "ray start --address='\K[^']+")
echo ">>> Using RAY_ADDRESS=${RAY_ADDRESS}"
ray status --address=${RAY_ADDRESS}

# ============ Environment ============
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============ Training ============
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir=${RAY_TMPDIR} \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=${BATCH_SIZE} \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=${N_VAL} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.reward_manager=multi_thread \
  probe.enable=False \
  probe.num_truncations=5 \
  probe.mc_samples=10 \
  probe.mc_max_tokens=32 \
  'probe.suffix= Thus, the final answer is: \\boxed{' \
  probe.num_splits=1 \
  probe.overconf_coeff=0.0 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
  "$@"
