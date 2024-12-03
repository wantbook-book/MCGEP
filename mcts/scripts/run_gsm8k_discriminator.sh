CUDA_VISIBLE_DEVICES=0 python run_src/do_discriminate.py \
    --model_ckpt microsoft/Phi-3-mini-4k-instruct \
    --root_dir /pubshare/fwk/code/rStar/run_outputs/GSM8K/Mistral-7B-v0.1/2024-11-19_15-43-35---[100q] \
    --dataset_name GSM8K \
    --note 100q
