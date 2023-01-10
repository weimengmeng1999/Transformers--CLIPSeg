#!/bin/bash -l
#SBATCH --output=/jmain02/home/J2AD019/exk01/%u/logs/%j.out
#SBATCH --job-name=openpsg
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

GPUS=8
PORT=29500

# Config
MODEL="clip-vit-base-patch32"
IMAGE_TYPE="image_path"
TEXT_TYPE="caption"
DATA_DIR="/jmain02/home/J2AD019/exk01/zxz35-exk01/data/data/transformers/data"
DATASET_NAME="ydshieh/coco_dataset_script"
DATASET_CONFIG_NAME="2017"
OUTPUT_DIR="./clip-vit-base-patch32-finetune-psg"

PYTHONPATH="./src":$PYTHONPATH \
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  examples/pytorch/contrastive-image-text/run_clip.py \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path openai/$MODEL \
  --data_dir $DATA_DIR \
  --dataset_name $DATASET_NAME \
  --dataset_config_name $DATASET_CONFIG_NAME \
  --image_column $IMAGE_TYPE \
  --caption_column $TEXT_TYPE \
  --max_seq_length 77 \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 10 \
  --learning_rate "5e-5" \
  --warmup_steps 0 \
  --weight_decay 0.1 \
  --overwrite_output_dir \
  --report_to none