#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=conv_6L_8-%N-%j.out    # %N for node name, %j for jobID model_conv_incp3_128.out

module load cuda cudnn python/3.5.2
source tensorflow3/bin/activate

python /home/rbbidart/breakHis/src/train_models_k.py /home/rbbidart/project/rbbidart/breakHis/mkfold_keras_8/fold1 /home/rbbidart/breakHis/output/conv_6L_8 100 8 conv_6L 100 8