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
source /zhome/31/c/147318/irishcream/bin/activate

cd /zhome/31/c/147318/DL_DDPM

JID=${LSB_JOBID}
python train.py MNIST 30 64
