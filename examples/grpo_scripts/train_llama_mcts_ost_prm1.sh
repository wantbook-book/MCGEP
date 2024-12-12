set -x 
# export CUDA_VISIBLE_DEVICES=4,5,6
wandb_token=b0255391060d68833e9b98941b9eb94fe770fbe4
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/pubshare/fwk/code/MCGEP"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray_mcts \
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
   --n_samples_per_prompt 16 \
   --value_head_prefix lm_head \
   --vocab_size 32000 \
   --save_steps 100 \
   --save_path /pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-mcts_ost_prm_test  \
   --micro_train_batch_size 8 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 16 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /pubshare/fwk/code/MCGEP/dataset/math/train.jsonl \
   --input_key problem \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --dataset_name MATH \
   --test_json_filename test_all \
   --note 10q \
   --data_root /pubshare/fwk/code/MCGEP/mcts/data \
   --prompts_root /pubshare/fwk/code/MCGEP/mcts/prompts \
   --run_outputs_root /pubshare/fwk/code/MCGEP/run_outputs \
   --num_rollouts 1 \
   --mcts_mode only_ost \
   --mcts_use_prm \
   --max_depth_allowed 20
# num_rollouts 似乎没有必要 >1 按我现在实现的


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
