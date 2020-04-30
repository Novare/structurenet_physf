#!/bin/bash
pyenv uninstall 3.6.5
pyenv install 3.6.5
pip install -r requirements.txt
pip install torch==1.4.0 torchvision==0.5.0
pip install torch-scatter==1.4.0
