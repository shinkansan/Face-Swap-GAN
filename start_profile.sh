#! /usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate simswap
cd /home/elba/elba_faceGAN
python -m cProfile -o fsgan.profile /home/elba/elba_faceGAN/gui.py
