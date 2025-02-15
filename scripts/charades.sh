VIDEO_PATH=/home/zyl/MeVTR_data_and_models/charades/Charades_v2_3
ANNO_PATH=/home/zyl/MeVTR_data_and_models/charades/annotation
OUTPUT_PATH=/home/zyl/TempMe/output_charades
PRETRAINED_PATH=/home/zyl/pretrained_clip

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --do_train 1 --workers 8 \
  --anno_path ${ANNO_PATH} --video_path ${VIDEO_PATH} --datatype charades \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH} \
  --batch_size 64 --batch_size_val 64
