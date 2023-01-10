#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=clip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144
#SBATCH --time=2-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

GPUS=1
PORT=29500

# Config
MODEL="clip-vit-base-patch16"
ROOT_DIR="/scratch/grp/grv_shi/cambridge-1/data/EndoVis2017/cropped_train"
# ROOT_DIR="/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_train"
SEG_TYPE="instruments"
TRAIN_TYPE="finetune_fc"
IMAGE_TYPE="image_path"
TEXT_TYPE="seg_class_name"
TRAIN_FILE="$ROOT_DIR/train_${SEG_TYPE}_clip.json"
VAL_FILE="$ROOT_DIR/train_${SEG_TYPE}_clip.json"
TEST_FILE="$ROOT_DIR/train_${SEG_TYPE}_clip.json"
OUTPUT_DIR="./$MODEL-cambridge1-$SEG_TYPE-$IMAGE_TYPE-$TEXT_TYPE-$TRAIN_TYPE"

PYTHONPATH="./src":$PYTHONPATH \
python examples/pytorch/contrastive-image-text/run_clip.py \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path openai/$MODEL \
  --image_column $IMAGE_TYPE \
  --caption_column $TEXT_TYPE \
  --train_file $TRAIN_FILE \
  --validation_file $VAL_FILE \
  --test_file $TEST_FILE \
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
  --report_to none \
  --freeze_vision_model \
  --freeze_text_model