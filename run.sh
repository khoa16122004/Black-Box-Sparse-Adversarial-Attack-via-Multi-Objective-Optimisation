#!/bin/bash

python run_all_batch.py \
    --runned_data_path model_evaluation_results/resnet18.json \
    --output_dir outdir \
    --model_name resnet18 \
    --eps 100 \
    --iterations 100 \
    --pc 0.1 \
    --pm 0.4 \
    --pop_size 100 \
    --zero_probability 0.3 \
    --max_dist 1e-5 \
    --p_size 2.0 \
    --tournament_size 4 \
    --explain_method simple_grad \
    --ig_steps 5 \
    --print_every 10
