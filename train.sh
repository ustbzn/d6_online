#！/bin/bash
GPUS=8
PORT=12098
CONFIG1=configs/sweeper/v3_C3_add_train_bj_C3_cam5lidar_voxelpoint50_250731.yaml
RUN_DIR1=experiments/v3_C3_add_train_bj_C3_cam5lidar_voxelpoint50_250731_4
CONFIG2=configs/mogo/v23_cutPasteNoise.yaml
RUN_DIR2=experiments/v23_cutPasteNoise
# CONFIG3=configs/mogo/v23_cutPasteNoise_onlycam.yaml
# RUN_DIR3=experiments/v23_cutPasteNoise_onlycam
MODEL1=$RUN_DIR1/latest.pth
MODEL2=$RUN_DIR2/latest.pth
# MODEL3=$RUN_DIR3/latest.pth

# torchrun --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train.py \
#     --config $CONFIG3 \
#     --run_dir $RUN_DIR3 \
#     --launcher pytorch

# torchrun --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/test.py \
#     --config $CONFIG3 \
#     --checkpoint $MODEL3 \
#     --rundir $RUN_DIR3 \
#     --launcher pytorch \
#     --eval bbox

# torchrun --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train.py \
#     --config $CONFIG1 \
#     --run_dir $RUN_DIR1 \
#     --launcher pytorch

# torchrun --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/test.py \
#     --config $CONFIG1 \
#     --checkpoint $MODEL1 \
#     --rundir $RUN_DIR1 \
#     --launcher pytorch \
#     --eval bbox

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    --config $CONFIG1 \
    --run_dir $RUN_DIR1 \
    --launcher pytorch

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py \
    --config $CONFIG1 \
    --checkpoint $MODEL1 \
    --rundir $RUN_DIR1 \
    --launcher pytorch \
    --eval bbox

# python /rbs/houjiayue/gpu2.py --gpus 4 --size 20000 --interval 0.01