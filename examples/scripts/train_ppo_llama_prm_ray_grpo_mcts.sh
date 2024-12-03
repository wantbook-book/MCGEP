set -x 
export CUDA_VISIBLE_DEVICES=4,5,6,7
wandb_token=b0255391060d68833e9b98941b9eb94fe770fbe4
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/pubshare/fwk/code/MCGEP"}' \
   -- python3 -m openrlhf.cli.train_ppo_prm_ray_mcts \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain /home/jovyan/share/LLMAgent/model/Llama-3.2-1B-Instruct \
   --reward_pretrain /pubshare/LLM/math-shepherd-mistral-7b-prm \
   --advantage_estimator group_norm \
   --n_samples_per_prompt 2 \
   --value_head_prefix lm_head \
   --vocab_size 32000 \
   --use_prm \
   --save_path /pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-porm_grpo_n2_mcts \
   --micro_train_batch_size 1 \
   --train_batch_size 1 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 1 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /pubshare/fwk/code/openRLHF_org2/dataset/math/train.jsonl \
   --input_key problem \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb b0255391060d68833e9b98941b9eb94fe770fbe4 \
   --wandb_run_name llama3-1b-porm_grpo_n2_mcts \
   --wandb_relogin True \
   --dataset_name MATH \
   --test_json_filename test_all \
   --note 10q \
   --num_rollouts 1 \
   --data_root /pubshare/fwk/code/openRLHF_org2/mcts/data \
   --prompts_root /pubshare/fwk/code/openRLHF_org2/mcts/prompts \
   --run_outputs_root /pubshare/fwk/code/openRLHF_org2/run_outputs \
   --mcts_mode full_actions

   # --save_tree \
   #  --model_ckpt mistralai/Mistral-7B-v0.1 \

   # --micro_train_batch_size 4 \
   # --train_batch_size 64 \
   # --micro_rollout_batch_size 16 \
   # --rollout_batch_size 512 \
   # --use_wandb ${wandb_token} \
# OpenRLHF/Llama-3-8b-rm-mixture
# /pubshare/LLM/math-shepherd-mistral-7b-prm
# --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward
   # --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
# --reward_pretrain /pubshare/LLM/math-shepherd-mistral-7b-prm \
# --micro_train_batch_size 8 \
   # --train_batch_size 128 \
   # --micro_rollout_batch_size 32 \
   # --rollout_batch_size 1024 \