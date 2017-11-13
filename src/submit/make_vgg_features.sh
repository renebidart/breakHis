#!/bin/bash

#SBATCH --account=def-aghodsib
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-24:00            # time (DD-HH:MM)
#SBATCH --output=%x-%j.out

module load cuda cudnn python/3.5.2
source tensorflow7/bin/activate

python /home/rbbidart/breakHis/src/make_vgg_features.py project/rbbidart/breakHis/mkfold_keras_8 project/rbbidart/breakHis/features/vgg 100
