#!/bin/bash

# MS Loss (BL)
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_blzzzzz/run_1 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_bl/run_2 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_bl/run_3 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_bl/run_4 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_bl/run_5 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512                                              

# ANU MS Loss (Ours)
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_1 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_2 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_3 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_4 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_5 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512

#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_fastap/run_1 LOSSES.NAME anu_fastap_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_fastap/run_2 LOSSES.NAME anu_fastap_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_fastap/run_3 LOSSES.NAME anu_fastap_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_fastap/run_4 LOSSES.NAME anu_fastap_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_fastap/run_5 LOSSES.NAME anu_fastap_loss

#for i in {1..5}
#do
#    CUDA_VISIBLE_DEVICES=1 python tools/main.py \
#        --cfg configs/example.yaml \
#        SAVE_DIR output_fastap_bl/run_$i
#done

