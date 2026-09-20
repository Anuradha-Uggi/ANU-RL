# Triplet Loss (BL)
CUDA_VISIBLE_DEVICES=0 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_bl/run_1/recalls.txt' \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_bl/run_1/best_model.pth'

CUDA_VISIBLE_DEVICES=0 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_bl/run_2/recalls.txt' \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_bl/run_2/best_model.pth'
                                              

CUDA_VISIBLE_DEVICES=0 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_bl/run_3/recalls.txt' \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_bl/run_3/best_model.pth'                                              
                                              
# Triplet Loss (BL)
CUDA_VISIBLE_DEVICES=0 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_anu/run_1/recalls.txt' \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_anu/run_1/best_model.pth'

CUDA_VISIBLE_DEVICES=0 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_anu/run_2/recalls.txt' \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_anu/run_2/best_model.pth'

CUDA_VISIBLE_DEVICES=0 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_anu/run_3/recalls.txt' \
                                              LOSSES.NAME anu_npairs_loss \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/output_npairs_loss_anu/run_3/best_model.pth'

"""
#ALL
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/ALL/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/ALL/run_1/best_model.pth'

python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/ALL/run_2/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/ALL/run_2/best_model.pth'
                                              
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/ALL/run_3/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/ALL/run_3/best_model.pth'
                                              

#Easiest                                              
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Easiest/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Easiest/run_1/best_model.pth'            
                                              

python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Easiest/run_2/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Easiest/run_2/best_model.pth'            

python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Easiest/run_3/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Easiest/run_3/best_model.pth'            
                                              
                                                                                                                                                                            
#Hardest
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Hardest/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Hardest/run_1/best_model.pth'       
                                              
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Hardest/run_2/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Hardest/run_2/best_model.pth'       
                                              
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Hardest/run_3/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/Hardest/run_3/best_model.pth'   
                                              

#BL
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_ms_loss_bl/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_ms_loss_bl/run_1/best_model.pth'       
                                              
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_ms_loss_bl/run_2/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_ms_loss_bl/run_2/best_model.pth'       
                                              
python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_ms_loss_bl/run_3/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_ms_loss_bl/run_3/best_model.pth'                                               
                                                                                                                                                   

"""

"""
############################################## ANU Loss
CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_1/best_model.pth'

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_2/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_2/best_model.pth'                                              
CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_3/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_3/best_model.pth'

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_4/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_4/best_model.pth'

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_5/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/train.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/cars196/cars196_metric_ms_structure/test.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss_DEBUG/run_5/best_model.pth' 
                                              
############################################# BL MS Loss
                                              


CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/train_labels.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/test_labels.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_1/best_model.pth'

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_2/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/train_labels.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/test_labels.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_2/best_model.pth'                                              

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_3/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/train_labels.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/test_labels.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_3/best_model.pth'

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_4/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/train_labels.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/test_labels.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_4/best_model.pth'

CUDA_VISIBLE_DEVICES=1 python validator.py --cfg configs/example.yaml \
                                              SAVE_DIR '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_5/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/train_labels.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/deepfashion-inshop-clothes-retrieval/test_labels.txt' \
                                              MODEL.HEAD.DIM 512 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/inshop_triplet_loss_bl/run_5/best_model.pth'
############################################# BL MS Loss """
