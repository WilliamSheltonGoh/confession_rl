#!/bin/bash

# 停止相关进程
ps -ef | grep -E 'verl|vllm|ray|python' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ray stop --force

# 安装依赖
pip install 'numpy<2.3' --quiet


# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export DEEPSEEK_API_KEY="sk-XXXXXXXXX"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"


export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_ARTIFACT_DIR=/root/autodl-tmp/wandb_artifacts
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /root/autodl-tmp/ray
mkdir -p /root/autodl-tmp/wandb
mkdir -p /root/autodl-tmp/wandb_artifacts
mkdir -p /root/autodl-tmp/wandb_cache
mkdir -p /root/autodl-tmp/wandb_data
mkdir -p /root/autodl-tmp/hf_cache
mkdir -p /root/autodl-tmp/checkpoints/verl_grpo_self_rl/self_rl_data



# 核心命令：所有配置参数作为 python 命令的参数，合并到一行
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train=False \
  data.train_files=/root/verl/data/kandk_data/train.parquet \
  data.val_files=/root/verl/data/kandk_data/test.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  custom_reward_function.path=/root/verl/kandkreward.py \
  custom_reward_function.name=compute_score \
  data.shuffle=False \
  actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-4B \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=8e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=6 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=3 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console","wandb"]' \
  trainer.project_name=verl_grpo_4kandk \
  trainer.experiment_name=kandk_test_325finaltry \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=6 \
  trainer.test_freq=3 \
  trainer.total_epochs=2 \
  trainer.default_local_dir=/root/autodl-tmp/checkpoints/verl_grpo_self_rl/self_rl_data
