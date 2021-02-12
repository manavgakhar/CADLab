#!/bin/bash

python train_densenet.py --arch 'resnext101_32x8d' --img_size 256 --crop_size 224 --epoch 1 --batch_size 32 --learning_rate 0.001 --gpu 0 --pretrained 'True'

exit 0
