#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=vgg16_ft_8-%N-%j.out    # %N for node name, %j for jobID model_conv_incp3_128.out

module load cuda cudnn python/3.5.2
source tensorflow4/bin/activate

python /home/rbbidart/breakHis/src/train_models_k.py /home/rbbidart/project/rbbidart/breakHis/mkfold_keras_8/fold1 /home/rbbidart/breakHis/output/vgg16_ft_8 100 8 vgg16_ft 100 8