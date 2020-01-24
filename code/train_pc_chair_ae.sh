#!/bin/bash
export CC=gcc
export CXX=g++
pyenv shell 3.6.5
bash scripts/pretrain_part_pc_ae_chair.sh
bash scripts/train_pc_ae_chair.sh
bash scripts/eval_recon_pc_ae_chair.sh
