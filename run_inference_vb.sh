#!/bin/bash



CUDA_VISIBLE_DEVICES=3 python inference.py -mc Experiments/tasnet/config.json \
    -dc config/inference/test_vb.json \
    -cp Experiments/tasnet/checkpoints/latest_model.tar \
    -dist output/tasnet