CUDA_VISIBLE_DEVICES=4 python run_src/do_generate.py \
    --dataset_name GSM8K \
    --test_json_filename test_all \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --note default \
    --num_rollouts 16
