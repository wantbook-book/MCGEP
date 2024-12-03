#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/pubshare/fwk/code/MCGEP
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_src/test_only_ost_reasoning.py \
    --dataset_name GSM8K \
    --test_json_filename test_all \
    --pretrain mistralai/Mistral-7B-v0.1 \
    --note 10q \
    --num_rollouts 16 \
    --end_idx 10 \
    --data_root /pubshare/fwk/code/MCGEP/mcts/data
    # --model_parallel
