#!/bin/bash
export CC=gcc
export CXX=g++
pyenv shell 3.6.5
bash scripts/pretrain_part_pc_vae_chair.sh
bash scripts/train_pc_vae_chair.sh
bash scripts/eval_gen_pc_vae_chair.sh
