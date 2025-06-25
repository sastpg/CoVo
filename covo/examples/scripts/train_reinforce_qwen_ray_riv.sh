set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf/CoVo/covo", "excludes":["dataset/", "run_outputs/"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --vllm_gpu_memory_utilization 0.5 \
   --colocate_actor_ref \
   --pretrain /path/to/Qwen2.5-7B-Instruct \
   --reward_pretrain /do_not_care \
   --remote_rm_url https://do/not/care \
   --save_path /path/to/qwen7b_instruct_riv \
   --ckpt_path /path/to/qwen7b_instruct_riv \
   --micro_train_batch_size 8 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 1e-4 \
   --prompt_data /openrlhf/CoVo/covo/dataset/train.jsonl \
   --eval_data /openrlhf/CoVo/covo/dataset/test.jsonl \
   --input_key problem \
   --label_key solution \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 200 \
   --eval_steps 20 \
   --save_hf_ckpt \
   --flash_attn \
   --logging_path /openrlhf/CoVo/covo/run_outputs/qwen_riv_log.jsonl \
   --enable_accuracy_filter \
   --save_output_path /openrlhf/CoVo/covo/run_outputs \
   --intrinsic_reward riv

# You could also try
#   --enable_curiosity
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# NOTE: You can use wandb to log the training process.
# --use_wandb YOUR_WANDB_KEY \
# --wandb_run_name qwen3b_instruct_riv \