# Config
MODEL="clip-vit-base-patch32"
IMAGE_TYPE="image_path"
TEXT_TYPE="caption"
TRAIN_FILE="/jmain02/home/J2AD019/exk01/zxz35-exk01/data/data/psg_clip/train_clip.json"
VAL_FILE="/jmain02/home/J2AD019/exk01/zxz35-exk01/data/data/psg_clip/test_clip.json"
TEST_FILE="/jmain02/home/J2AD019/exk01/zxz35-exk01/data/data/psg_clip/test_clip.json"
OUTPUT_DIR="./clip-vit-base-patch32-finetune-psg"

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
  --report_to none