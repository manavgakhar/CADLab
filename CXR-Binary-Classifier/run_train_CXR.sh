#!/bin/bash

python train_densenet.py --img_size 256 --crop_size 224 --batch_size 32 --learning_rate 0.001 --gpu 0 --pretrained 'True'

exit 0
