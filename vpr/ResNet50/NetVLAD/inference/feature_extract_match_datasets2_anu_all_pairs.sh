'''
####### Amstertime ######
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --output_features_dir=./ANU-All-Pairs/amstertime/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt"


python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/amstertime_db_names.txt --query_input_features_dir=./ANU-All-Pairs/amstertime/ --index_input_features_dir=./ANU-All-Pairs/amstertime/ --result_save_folder=./ANU-All-Pairs/amstertime/ 


####### ESSEX3IN1 #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --output_features_dir=./ANU-All-Pairs/essex3in1/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_query_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/essex3in1_db_names.txt --query_input_features_dir=./ANU-All-Pairs/essex3in1/ --index_input_features_dir=./ANU-All-Pairs/essex3in1/ --result_save_folder=./ANU-All-Pairs/essex3in1/ 


####### GardensPointWalking right #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --output_features_dir=./ANU-All-Pairs/gardens_right/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_right_names.txt --query_input_features_dir=./ANU-All-Pairs/gardens_right/ --index_input_features_dir=./ANU-All-Pairs/gardens_right/ --result_save_folder=./ANU-All-Pairs/gardens_right/ 


####### GardensPointWalking left #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --output_features_dir=./ANU-All-Pairs/gardens_left/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/GardensPointWalking_db_left_names.txt --query_input_features_dir=./ANU-All-Pairs/gardens_left/ --index_input_features_dir=./ANU-All-Pairs/gardens_left/ --result_save_folder=./ANU-All-Pairs/gardens_left/ 


####### SPEDTest #####
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --output_features_dir=./ANU-All-Pairs/sped/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/sped_db_names.txt --query_input_features_dir=./ANU-All-Pairs/sped/ --index_input_features_dir=./ANU-All-Pairs/sped/ --result_save_folder=./ANU-All-Pairs/sped/ '''


########################## Additional datasets ####################################
####### Eynsham
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --output_features_dir=./ANU-All-Pairs/eynsham/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/eynsham_db_names.txt --query_input_features_dir=./ANU-All-Pairs/eynsham/ --index_input_features_dir=./ANU-All-Pairs/eynsham/ --result_save_folder=./ANU-All-Pairs/eynsham/

######## St_Lucia
#python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --output_features_dir=./ANU-All-Pairs/st_lucia/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
#python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/st_lucia_db_names.txt --query_input_features_dir=./ANU-All-Pairs/st_lucia/ --index_input_features_dir=./ANU-All-Pairs/st_lucia/ --result_save_folder=./ANU-All-Pairs/st_lucia/ 

######## SVOX
#python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/svox/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 
 
#python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/svox/ --index_input_features_dir=./ANU-All-Pairs/svox/ --result_save_folder=./ANU-All-Pairs/svox/

########## SVOX Night
'''
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/svox/night/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 

python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_night_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/svox/night/ --index_input_features_dir=./ANU-All-Pairs/svox/ --result_save_folder=./ANU-All-Pairs/svox/night/

############# SVOX Overcast
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/svox/overcast/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 

python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_overcast_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/svox/overcast/ --index_input_features_dir=./ANU-All-Pairs/svox/ --result_save_folder=./ANU-All-Pairs/svox/overcast/

############ SVOX Rain
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/svox/rain/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 

python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_rain_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/svox/rain/ --index_input_features_dir=./ANU-All-Pairs/svox/ --result_save_folder=./ANU-All-Pairs/svox/rain/

############ SVOX Snow
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/svox/snow/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 


python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_snow_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/svox/snow/ --index_input_features_dir=./ANU-All-Pairs/svox/ --result_save_folder=./ANU-All-Pairs/svox/snow/

############ SVOX Sun
python feature_extract.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --output_features_dir=./ANU-All-Pairs/svox/sun/ --model_path="./../ANU-All-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(17)_step(11268)_R1[0.8986]_R5[0.9462].ckpt" 


python feature_match.py --query_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_qry_sun_names.txt --index_file_path=/data/anuradha/Used-Datasets-WACV-25/dataset_names/svox_db_names.txt --query_input_features_dir=./ANU-All-Pairs/svox/sun/ --index_input_features_dir=./ANU-All-Pairs/svox/ --result_save_folder=./ANU-All-Pairs/svox/sun/ '''
