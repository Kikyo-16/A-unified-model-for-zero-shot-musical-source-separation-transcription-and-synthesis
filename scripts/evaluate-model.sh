cuda=$1
model_name=$2
epoch=$3
model_folder=$4
CUDA_VISIBLE_DEVICES=$cuda python evaluate.py --model_name=$model_name --model_path=$model_folder/params_epoch-$epoch.pkl \
																							--evaluation_folder=evaluation --epoch=$epoch
