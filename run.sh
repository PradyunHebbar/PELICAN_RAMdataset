#!/bin/bash

# WORKS
module load python
module load pytorch/2.3.1
conda activate pelican


python3 train_pelican_classifier.py --datadir=../atlas_data --target=is_signal --nobj=80 --nobj-avg=56 --num-epoch=30 --num-train=10000000 --num-valid=1000000 --num-test=1000000  --batch-size=128 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005