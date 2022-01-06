cuda=$1
model_name=$2
model_folder=$3
CUDA_VISIBLE_DEVICES=$cuda python train.py --model_name=$model_name --model_folder=$model_folder --epoch=200
