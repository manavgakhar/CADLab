#!/bin/bash

python test_densenet.py --arch 'resnext101_32x8d' --img_size 256 --crop_size 224 --batch_size 64 --learning_rate 0.001 --gpu 0 --test_labels 'att'

exit 0
