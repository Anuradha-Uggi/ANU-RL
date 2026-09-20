#!/bin/bash
# ms Loss (BL)

#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
#                                              SAVE_DIR output_ms_loss_bl/run_1 \
#                                              LOSSES.NAME ms_loss \
#                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
#                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
#                                              MODEL.HEAD.DIM 64 \
#                                              MODEL.PRETRAIN 'imagenet'
                                              
# ANU ms Loss (Ours)



# tripelt Loss (BL)
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_loss_bl/run_1 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_loss_bl/run_2 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_loss_bl/run_3 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_loss_bl/run_4 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet' 
                                              
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_fastap_loss_bl/run_5 \
                                              LOSSES.NAME fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet' 

# ANU triplet Loss (Ours)
CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_1 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_2 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_3 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_4 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'

CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
                                              SAVE_DIR output_anu_fastap_loss/run_5 \
                                              LOSSES.NAME anu_fastap_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN 'imagenet'




#sizes=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#for i in "${!sizes[@]}"
#do 
#   dataS="${sizes[$i]}"
#   echo "Index: $i, Value: $dataS"
   
#   CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml \
#                                              SAVE_DIR output_anu_ms_loss/run_$dataS \
#                                              LOSSES.NAME anu_ms_loss \
#                                              LOSSES.REG_LAMBDA $dataS \
#                                              DATA.TRAIN_IMG_SOURCE "resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt" \
#                                              DATA.TEST_IMG_SOURCE "resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt" \
#                                              MODEL.HEAD.DIM 64 \
#                                              MODEL.PRETRAIN "imagenet"
                                              
#done


#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_ms_loss/run_1 LOSSES.NAME anu_ms_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_ms_loss/run_2 LOSSES.NAME anu_ms_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_ms_loss/run_3 LOSSES.NAME anu_ms_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_ms_loss/run_4 LOSSES.NAME anu_ms_loss
#CUDA_VISIBLE_DEVICES=1 python tools/main.py --cfg configs/example.yaml SAVE_DIR output_anu_ms_loss/run_5 LOSSES.NAME anu_ms_loss

#for i in {1..5}
#do
#    CUDA_VISIBLE_DEVICES=1 python tools/main.py \
#        --cfg configs/example.yaml \
#        SAVE_DIR output_ms_loss_bl/run_$i
#done

