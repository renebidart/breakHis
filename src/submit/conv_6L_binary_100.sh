#!/bin/bash

#SBATCH --account=def-aghodsib
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=%x-%j.out

module load cuda cudnn python/3.5.2
source tensorflow/bin/activate

python /home/rbbidart/breakHis/src/test_models_aug.py project/rbbidart/breakHis/by_patient /home/rbbidart/breakHis/output/conv_6L_binary_100 50 64 256 conv_6L 100 True
