for dataS in 112 320

do 
####### P30k ###### T2
   #python feature_extract_new.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_index.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/p30k/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='p30k' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_index.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/p30k/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/p30k/ --ground_truth_path=/home/anuradha/dataset_names/dataset_gt_files/pitts30k_test.npz --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/p30k/ --data_name='p30k' --inp_dim=$dataS



####### nordland ##### T2============================================================
   #python feature_extract_new.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_index.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/nordland/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='nordland' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_index.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/nordland/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/nordland/ --ground_truth_path=/home/anuradha/dataset_names/dataset_gt_files/nordland.npz --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/nordland/ --data_name='nordland' --inp_dim=$dataS



####### msls ##### T2============================================================
   #python feature_extract_new.py --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/msls/ --msls --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='msls' --inp_dim=$dataS 
 
   python feature_match.py --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/msls/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/msls/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/msls/ --msls --data_name='msls' --inp_dim=$dataS


###### Tokyo 
####### Tokyo ####### T2 ============================================================
   python feature_extract_new.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_index.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/tokyo/ --dataset_root_dir_1=/data/anuradha/Anus-PhD/TokyoV3/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='tokyo' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_index.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/tokyo/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/tokyo/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/tokyo/ --ground_truth_path=/home/anuradha/dataset_names/dataset_gt_files/tokyo247.npz --dataset_root_dir=/data/anuradha/Anus-PhD/TokyoV3/ --data_name='tokyo' --inp_dim=$dataS

## ===========================================================================================================================
#================================================================================================================================
##================================= Datasets 2 =================================================

####### Amstertime ######
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/amstertime/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='amstertime' --inp_dim=$dataS


   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/amstertime/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/amstertime/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/amstertime/ --data_name='amstertime' --inp_dim=$dataS


####### ESSEX3IN1 #####
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/essex3in1/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='essex3in1' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/essex3in1/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/essex3in1/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/essex3in1/ --data_name='essex3in1' --inp_dim=$dataS


####### GardensPointWalking right #####
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_right/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='gardens_right' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_right/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_right/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_right/ --data_name='gardens_right' --inp_dim=$dataS


####### GardensPointWalking left #####
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_left/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='gardens_left' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_left/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_left/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/gardens_left/ --data_name='gardens_left' --inp_dim=$dataS


####### SPEDTest #####
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/sped/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='sped' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/sped/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/sped/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/sped/ --data_name='sped' --inp_dim=$dataS

########################## Additional datasets ####################################

######## St_Lucia

   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/st_lucia/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='st_lucia' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/st_lucia/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/st_lucia/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/st_lucia/ --data_name='st_lucia' --inp_dim=$dataS


####### Eynsham
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/eynsham/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='eynsham' --inp_dim=$dataS
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/eynsham/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/eynsham/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/eynsham/ --data_name='eynsham' --inp_dim=$dataS


######## SVOX
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --inp_dim=$dataS 
 
   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --data_name='svox' --inp_dim=$dataS


########## SVOX Night

   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/night/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='svoxvar' --inp_dim=$dataS

   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/night/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/night/ --data_name='svoxvar' --inp_dim=$dataS

############# SVOX Overcast
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/overcast/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='svoxvar' --inp_dim=$dataS

   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/overcast/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/overcast/ --data_name='svoxvar' --inp_dim=$dataS

############ SVOX Rain
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/rain/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='svoxvar' --inp_dim=$dataS

   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/rain/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/rain/ --data_name='svoxvar' --inp_dim=$dataS

############ SVOX Snow
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/snow/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='svoxvar' --inp_dim=$dataS


   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/snow/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/snow/ --data_name='svoxvar' --inp_dim=$dataS

############ SVOX Sun
   #python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/sun/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(34)_step(21910)_R1[0.8923]_R5[0.9460].ckpt" --data_name='svoxvar' --inp_dim=$dataS


   python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/sun/ --index_input_features_dir=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/ --result_save_folder=./ANU-All-Pairs/TMLR-Rev/$dataSx$dataS-dim/svox/sun/ --data_name='svoxvar' --inp_dim=$dataS  #''' 


done

