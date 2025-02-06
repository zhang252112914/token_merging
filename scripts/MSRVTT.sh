VIDEO_PATH=/home/zyl/Xpool/dataset/msrvtt/MSRVTT_Videos
ANNO_PATH=/home/zyl/Xpool/dataset/msrvtt/MSRVTT
OUTPUT_PATH=/home/zyl/TempMe/output
PRETRAINED_PATH=/home/zyl/pretrained_clip

CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --do_train 1 --workers 8 \
  --anno_path ${ANNO_PATH} --video_path ${VIDEO_PATH} --datatype msrvtt \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH}
