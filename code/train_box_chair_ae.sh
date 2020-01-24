#!/bin/bash
export CC=gcc
export CXX=g++
pyenv shell 3.6.5
bash scripts/train_box_ae_chair.sh
bash scripts/eval_recon_box_ae_chair.sh
