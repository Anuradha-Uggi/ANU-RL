#!/bin/bash

# Triplet Loss (BL)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_1 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'


CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_2 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_3 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                                                                            
                                              
                                              
# Triplet Loss (BL)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_anu/run_1 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'

# Triplet Loss (BL)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_anu/run_2 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_anu/run_3 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE = 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'

"""                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_2 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_3 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_4 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_npairs_loss_bl/run_5 \
                                              LOSSES.NAME npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'                                             

"""



"""
# ANU Triplet Loss (Ours)
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_npairs_loss/run_1 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_npairs_loss/run_2 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_npairs_loss/run_3 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_npairs_loss/run_4 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_npairs_loss/run_5 \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/CUB_200_2011/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/CUB_200_2011/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN 'imagenet'



#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_1 LOSSES.NAME anu_npairs_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_2 LOSSES.NAME anu_npairs_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_3 LOSSES.NAME anu_npairs_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_4 LOSSES.NAME anu_npairs_loss
#CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_triplet/run_5 LOSSES.NAME anu_npairs_loss

#for i in {1..5}
#do
#    CUDA_VISIBLE_DEVICES=0 python tools/main.py \
#        --cfg configs/example.yaml \
#        SAVE_DIR output_triplet_bl/run_$i
#done
"""
