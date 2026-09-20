'''
####### Amstertime ######
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --output_features_dir=./LOGS-MSim-BL/amstertime/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt"


python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/amstertime/ --index_input_features_dir=./LOGS-MSim-BL/amstertime/ --result_save_folder=./LOGS-MSim-BL/amstertime/ 


####### ESSEX3IN1 #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --output_features_dir=./LOGS-MSim-BL/essex3in1/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/essex3in1/ --index_input_features_dir=./LOGS-MSim-BL/essex3in1/ --result_save_folder=./LOGS-MSim-BL/essex3in1/ 


####### GardensPointWalking right #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --output_features_dir=./LOGS-MSim-BL/gardens_right/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --query_input_features_dir=./LOGS-MSim-BL/gardens_right/ --index_input_features_dir=./LOGS-MSim-BL/gardens_right/ --result_save_folder=./LOGS-MSim-BL/gardens_right/ 


####### GardensPointWalking left #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --output_features_dir=./LOGS-MSim-BL/gardens_left/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --query_input_features_dir=./LOGS-MSim-BL/gardens_left/ --index_input_features_dir=./LOGS-MSim-BL/gardens_left/ --result_save_folder=./LOGS-MSim-BL/gardens_left/ 


####### SPEDTest #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --output_features_dir=./LOGS-MSim-BL/sped/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/sped/ --index_input_features_dir=./LOGS-MSim-BL/sped/ --result_save_folder=./LOGS-MSim-BL/sped/
'''

########################## Additional datasets ####################################
####### Eynsham
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --output_features_dir=./LOGS-MSim-BL/eynsham/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/eynsham/ --index_input_features_dir=./LOGS-MSim-BL/eynsham/ --result_save_folder=./LOGS-MSim-BL/eynsham/

######## St_Lucia
#python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --output_features_dir=./LOGS-MSim-BL/st_lucia/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
#python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/st_lucia/ --index_input_features_dir=./LOGS-MSim-BL/st_lucia/ --result_save_folder=./LOGS-MSim-BL/st_lucia/ 

######## SVOX
#python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./LOGS-MSim-BL/svox/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 
 
#python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/svox/ --index_input_features_dir=./LOGS-MSim-BL/svox/ --result_save_folder=./LOGS-MSim-BL/svox/

########## SVOX Night
'''
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./LOGS-MSim-BL/svox/night/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 

python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/svox/night/ --index_input_features_dir=./LOGS-MSim-BL/svox/ --result_save_folder=./LOGS-MSim-BL/svox/night/

############# SVOX Overcast
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./LOGS-MSim-BL/svox/overcast/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 

python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/svox/overcast/ --index_input_features_dir=./LOGS-MSim-BL/svox/ --result_save_folder=./LOGS-MSim-BL/svox/overcast/

############ SVOX Rain
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./LOGS-MSim-BL/svox/rain/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 

python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/svox/rain/ --index_input_features_dir=./LOGS-MSim-BL/svox/ --result_save_folder=./LOGS-MSim-BL/svox/rain/

############ SVOX Snow
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./LOGS-MSim-BL/svox/snow/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 


python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/svox/snow/ --index_input_features_dir=./LOGS-MSim-BL/svox/ --result_save_folder=./LOGS-MSim-BL/svox/snow/

############ SVOX Sun
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./LOGS-MSim-BL/svox/sun/ --model_path="./../LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(39)_step(25040)_R1[0.8999]_R5[0.9494].ckpt" 


python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./LOGS-MSim-BL/svox/sun/ --index_input_features_dir=./LOGS-MSim-BL/svox/ --result_save_folder=./LOGS-MSim-BL/svox/sun/ '''
