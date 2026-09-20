#!/bin/bash

# MS Loss (BL)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_ms_loss_bl/run_1 \
                                              LOSSES.NAME ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_ms_loss_bl/run_2 \
                                              LOSSES.NAME ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_ms_loss_bl/run_3 \
                                              LOSSES.NAME ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_ms_loss_bl/run_4 \
                                              LOSSES.NAME ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_ms_loss_bl/run_5 \
                                              LOSSES.NAME ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'                                             

# ANU MS Loss (Ours)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss/run_1 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss/run_2 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss/run_3 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss/run_4 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_ms_loss/run_5 \
                                              LOSSES.NAME anu_ms_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'








# Triplet Loss (BL)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_triplet_loss_bl/run_1 \
                                              LOSSES.NAME triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_triplet_loss_bl/run_2 \
                                              LOSSES.NAME triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_triplet_loss_bl/run_3 \
                                              LOSSES.NAME triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_triplet_loss_bl/run_4 \
                                              LOSSES.NAME triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_triplet_loss_bl/run_5 \
                                              LOSSES.NAME triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'                                             

# ANU Triplet Loss (Ours)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_triplet_loss/run_1 \
                                              LOSSES.NAME anu_triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_triplet_loss/run_2 \
                                              LOSSES.NAME anu_triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_triplet_loss/run_3 \
                                              LOSSES.NAME anu_triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_triplet_loss/run_4 \
                                              LOSSES.NAME anu_triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_triplet_loss/run_5 \
                                              LOSSES.NAME anu_triplet_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'



#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_1 LOSSES.NAME anu_triplet_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_2 LOSSES.NAME anu_triplet_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_3 LOSSES.NAME anu_triplet_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_4 LOSSES.NAME anu_triplet_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_5 LOSSES.NAME anu_triplet_loss

#for i in {1..5}
#do
#    CUDA_VISIBLE_DEVICES=0 python tools/main.py \
#        --cfg configs/example.yaml \
#        SAVE_DIR output_triplet_bl/run_$i
#done

