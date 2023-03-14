#!/bin/bash



CUDA_VISIBLE_DEVICES=0 python inference.py -mc Experiments/tasnet/config.json \
    -dc config/inference/test_vb.json \
    -cp Experiments/tasnet/checkpoints/best_model.tar \
    -dist output/tasnet_sisnr