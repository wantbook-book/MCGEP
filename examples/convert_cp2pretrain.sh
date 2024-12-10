set -x 
# export CUDA_VISIBLE_DEVICES=4,5,6
wandb_token=b0255391060d68833e9b98941b9eb94fe770fbe4
CHECKPOINT_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-porm_grpo_math_instruct/checkpoints/_actor"
CHECKPOINT_DIR=("${CHECKPOINT_ROOT}/global_step600" "${CHECKPOINT_ROOT}/global_step700" "${CHECKPOINT_ROOT}/global_step800")
CHECKPOINT_NAMES=("global_step600" "global_step700" "global_step800")
PRETRAIN_MODEL="/home/jovyan/share/LLMAgent/model/Llama-3.2-1B-Instruct"
# for checkpoint in "${CHECKPOINT_DIR[@]}"; do
for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{"working_dir": "/pubshare/fwk/code/MCGEP"}' \
        -- python3 -m openrlhf.cli.train_ppo_convert_cp2pretrain \
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
        --pretrain ${PRETRAIN_MODEL} \
        --reward_pretrain /pubshare/LLM/math-shepherd-mistral-7b-prm \
        --advantage_estimator group_norm \
        --n_samples_per_prompt 4 \
        --value_head_prefix lm_head \
        --vocab_size 32000 \
        --save_steps 100 \
        --save_path "${CHECKPOINT_ROOT}/${checkpoint_name}_pretrain"  \
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
        --num_rollouts 4 \
        --data_root /pubshare/fwk/code/MCGEP/mcts/data \
        --prompts_root /pubshare/fwk/code/MCGEP/mcts/prompts \
        --run_outputs_root /pubshare/fwk/code/MCGEP/run_outputs \
        --mcts_mode only_ost \
        --max_depth_allowed 20
done
