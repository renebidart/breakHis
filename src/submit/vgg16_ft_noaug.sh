#!/bin/bash

#SBATCH --account=def-aghodsib
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-24:00            # time (DD-HH:MM)
#SBATCH --output=%x-%j.out

module load cuda cudnn python/3.5.2
source tensorflow/bin/activate

python /home/rbbidart/breakHis/src/test_models_noaug.py project/rbbidart/breakHis/by_patient /home/rbbidart/breakHis/output/vgg16_ft_noaug 50 64 224 vgg16_ft