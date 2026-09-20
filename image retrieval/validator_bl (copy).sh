
############################################## ANU Loss
"""
CUDA_VISIBLE_DEVICES=1 python validator_bl.py --cfg configs/example.yaml \
                                              SAVE_DIR './Experiments-SOP-Training-Data/64-dim/TripletLoss/output_anu_triplet_loss/run_5/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/Experiments-SOP-Training-Data/64-dim/TripletLoss/output_anu_triplet_loss/run_5/best_model.pth'



############################################# BL MS Loss
CUDA_VISIBLE_DEVICES=1 python validator_bl.py --cfg configs/example.yaml \
                                              SAVE_DIR './Experiments-SOP-Training-Data/64-dim/MSimLoss/output_ms_loss_bl/run_5/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/Experiments-SOP-Training-Data/64-dim/MSimLoss/output_ms_loss_bl/run_5/best_model.pth' 
                                              


CUDA_VISIBLE_DEVICES=1 python validator_bl.py --cfg configs/example.yaml \
                                              SAVE_DIR './Experiments-SOP-Training-Data/64-dim/MSimLoss/output_anu_ms_loss/run_1/recalls.txt' \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN '/data/anuradha/WACV25_Reviews/IR/From_Sindhu/Experiments-SOP-Training-Data/64-dim/MSimLoss/output_anu_ms_loss/run_1/best_model.pth'                                              
                                              

                                  
                                  
CUDA_VISIBLE_DEVICES=0 python validator_bl.py --cfg configs/example.yaml \
                                              SAVE_DIR ./output_anu_ms_loss/run-ms-term/recalls.txt \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN /data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss/run-ms-term/best_model.pth                                  
""" 
                                              

sizes=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#sizes=(0.9 1.0)
for i in "${!sizes[@]}"
do 
   dataS="${sizes[$i]}"
   echo "Index: $i, Value: $dataS"                        
                                              
############################################## ANU MS Loss
   CUDA_VISIBLE_DEVICES=0 python validator_bl.py --cfg configs/example.yaml \
                                              SAVE_DIR ./output_anu_ms_loss-T3/run_$dataS/recalls.txt \
                                              DATA.TRAIN_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_train_upd.txt' \
                                              DATA.TEST_IMG_SOURCE 'resource/datasets/Stanford_Online_Products/Ebay_test_upd.txt' \
                                              MODEL.HEAD.DIM 64 \
                                              MODEL.PRETRAIN /data/anuradha/WACV25_Reviews/IR/From_Sindhu/output_anu_ms_loss-T3/run_$dataS/best_model.pth

done


