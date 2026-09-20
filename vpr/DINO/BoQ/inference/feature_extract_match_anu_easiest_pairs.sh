
####### P30k ###### T2
python feature_extract_new.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_index.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/p30k/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='p30k' --inp_dim=224
 
python feature_match.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/pitts30k_imageNames_index.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/p30k/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/p30k/ --ground_truth_path=/home/anuradha/dataset_names/dataset_gt_files/pitts30k_test.npz --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/p30k/ --data_name='p30k' --inp_dim=224



####### nordland ##### T2============================================================
python feature_extract_new.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_index.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/nordland/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='nordland' --inp_dim=224
 
python feature_match.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/nordland_imageNames_index.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/nordland/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/nordland/ --ground_truth_path=/home/anuradha/dataset_names/dataset_gt_files/nordland.npz --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/nordland/ --data_name='nordland' --inp_dim=224



####### msls ##### T2============================================================
python feature_extract_new.py --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/msls/ --msls --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='msls' --inp_dim=224
 
python feature_match.py --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/msls/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/msls/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/msls/ --msls --data_name='msls' --inp_dim=224


###### Tokyo 
####### Tokyo ####### T2 ============================================================
python feature_extract_new.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_index.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/tokyo/ --dataset_root_dir_1=/data/anuradha/Anus-PhD/TokyoV3/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='tokyo' --inp_dim=224
 
python feature_match.py --query_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_query.txt --index_file_path=/home/anuradha/dataset_names/dataset_imagenames/tokyo247_imageNames_index.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/tokyo/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/tokyo/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/tokyo/ --ground_truth_path=/home/anuradha/dataset_names/dataset_gt_files/tokyo247.npz --dataset_root_dir=/data/anuradha/Anus-PhD/TokyoV3/ --data_name='tokyo' --inp_dim=224


## ===========================================================================================================================
#================================================================================================================================
##================================= Datasets 2 =================================================


####### Amstertime ######
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/amstertime/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='amstertime' --inp_dim=224


python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/amstertime/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/amstertime/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/amstertime/ --data_name='amstertime' --inp_dim=224


####### ESSEX3IN1 #####
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/essex3in1/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='essex3in1' --inp_dim=224
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/essex3in1/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/essex3in1/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/essex3in1/ --data_name='essex3in1' --inp_dim=224


####### GardensPointWalking right #####
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_right/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='gardens_right' --inp_dim=224
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_right/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_right/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_right/ --data_name='gardens_right' --inp_dim=224


####### GardensPointWalking left #####
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_left/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='gardens_left' --inp_dim=224
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_left/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_left/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/gardens_left/ --data_name='gardens_left' --inp_dim=224


####### SPEDTest #####
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/sped/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='sped' --inp_dim=224
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/sped/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/sped/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/sped/ --data_name='sped' --inp_dim=224

########################## Additional datasets ####################################

######## St_Lucia

python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/st_lucia/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='st_lucia' --inp_dim=224
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/st_lucia/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/st_lucia/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/st_lucia/ --data_name='st_lucia' --inp_dim=224


####### Eynsham
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/eynsham/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='eynsham' --inp_dim=224
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/eynsham/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/eynsham/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/eynsham/ --data_name='eynsham' --inp_dim=224


######## SVOX
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --inp_dim=224 
 
python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --data_name='svox' --inp_dim=224


########## SVOX Night

python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/night/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='svoxvar' --inp_dim=224

python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/night/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/night/ --data_name='svoxvar' --inp_dim=224

############# SVOX Overcast
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/overcast/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='svoxvar' --inp_dim=224

python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/overcast/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/overcast/ --data_name='svoxvar' --inp_dim=224

############ SVOX Rain
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/rain/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='svoxvar' --inp_dim=224

python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/rain/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/rain/ --data_name='svoxvar' --inp_dim=224

############ SVOX Snow
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/snow/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='svoxvar' --inp_dim=224


python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/snow/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/snow/ --data_name='svoxvar' --inp_dim=224

############ SVOX Sun
python feature_extract_new.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/sun/ --model_path="./../ANU-Easiest-Pairs/TMLR-Rev/seed-3/resnet50/version_0/checkpoints/epoch[28]_R@1[0.8392]_R@5[0.8959].ckpt" --data_name='svoxvar' --inp_dim=224


python feature_match.py --query_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Anus-PhD/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/sun/ --index_input_features_dir=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/ --result_save_folder=./ANU-Easiest-Pairs/TMLR-Rev/seed3/svox/sun/ --data_name='svoxvar' --inp_dim=224 #''' 




