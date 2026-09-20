#!/bin/bash

# MS Loss (BL)
# ANU MS Loss (Ours)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss_DEBUG/run_1 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss_DEBUG/run_2 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'                                               
                                               
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss_DEBUG/run_3 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss_DEBUG/run_4 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'                                              
                                             
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss_DEBUG/run_5 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'                                              
