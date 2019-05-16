#!/usr/bin/env bash
export PYTHONIOENCODING=utf-8
# preprocessing and train
nohup python -u main.py --config ./configs/word.yml > nohup.out 2>&1 &

# train only
# CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml --train

# test
#CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml --test