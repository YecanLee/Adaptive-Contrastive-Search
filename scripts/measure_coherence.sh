#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: No test path provided. Please provide a test path as an argument."
    echo "Usage: $0 <test_path>"
    exit 1
fi

TEST_PATH="$1"

CUDA_VISIBLE_DEVICES=0 python ../measure_coherence.py \
    --opt_model_name facebook/opt-2.7b \
    --test_path "$TEST_PATH"