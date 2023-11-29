#!/bin/bash

#BSUB -q gpuv100 
#BSUB -J train
#BSUB -o outs/train_%J.out
#BSUB -n 1
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 60
#BSUB -gpu "num=1:mode=exclusive_process"

module load python3/3.11.3
module load cuda/12.1.1
source ~/irishcream/bin/activate

JID=${LSB_JOBID}
python train.py CIFAR10 3 128
