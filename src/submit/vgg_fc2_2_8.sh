#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=vgg_fc2_2_8-%N-%j.out    # %N for node name, %j for jobID

module load cuda cudnn python/3.5.2
source tensorflow6/bin/activate

python /home/rbbidart/breakHis/src/train_vgg_pre.py /home/rbbidart/project/rbbidart/breakHis/mkfold_keras_8/fold1 /home/rbbidart/breakHis_out/vgg_pre/vgg_fc2_8 100 16 vgg16_fc2 100 8
python /home/rbbidart/breakHis/src/train_vgg_pre.py /home/rbbidart/project/rbbidart/breakHis/mkfold_keras_2/fold1 /home/rbbidart/breakHis_out/vgg_pre/vgg_fc2_2 100 16 vgg16_fc2 100 2