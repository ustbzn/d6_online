function SEND_MSG(){
if [ $? -ne 0 ]; then
    result=$(curl -s http://quan.suning.com/getSysTime.do)
    datetime=${result:13:19}
    msg="<font color='warning'>$1 Failed.</font> \nContainer: jjy_8GPU \n<font color='comment'>$datetime</font>"
else
    result=$(curl -s http://quan.suning.com/getSysTime.do)
    datetime=${result:13:19}
    msg="$1 .  \nContainer: jjy_8GPU \n<font color='comment'>$datetime</font>"
fi
a=$(curl -s 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=bea27727-0a86-4b37-8efe-952842068233' \
   -H 'Content-Type: application/json' \
   -d '
   {
        "msgtype": "markdown",
        "markdown": {
            "content": "'"$msg"'"
         }
   }')
}

function CLEAN_GPU(){
  echo "Cleaning Nvidia Occupied."
  ps -ef|grep python |grep -v grep|awk '{print $2}'|xargs kill -9
  sleep 1
  nvidia-smi | grep MiB
  sleep 1
}

# CLEAN_GPU
MSG1="v3_mixB2C1_0724"
GPUS=8
PORT=11985
CONFIG=configs/sweeper/v3_C1_mixB2_45kllm35k_cam7lidar_voxelpoint20_250724.yaml
RUN_DIR=runs/v3_mixB2C1_0724
EVAL_DIR=evaluate/v3_mixB2C1_0724
MODEL=$RUN_DIR/epoch_30.pth
# ps -ef|grep python |grep -v grep|awk '{print $2}'|xargs kill -9
SEND_MSG "Start Run ${MSG1}"
# bash tools/dist_train.sh\
#     configs/mogo/v1_C3_mixB2_cam5lidar_volc_250418.yaml\
#     8 1 0 101.126.84.179 49457 --run-dir runs/v1_mix_0418 > log/v1_mix_0418_train.txt 2>&1
# bash tools/dist_train.sh configs/mogo/falcon_v4_77baseline_SENetFuser_epoch30.yaml 4 1 0 10.0.12.40
# python tools/train.py --config configs/mogo/v1_C3_mixB2_cam5lidar_volc_250418.yaml --run-dir runs/v1_mix_0418 > log/v1_mix_0418_train.txt 2>&1
torchrun --nproc_per_node=$GPUS \
      --master_port=$PORT \
      tools/train.py \
      --config $CONFIG \
      --run_dir $RUN_DIR \
      --launcher pytorch  > log/v3_mixB2C1_0724_train.txt 2>&1

# 检查训练命令是否成功执行
if [ $? -eq 0 ]; then
    SEND_MSG "${MSG1} train finish"
    sleep 30

    # python tools/test.py --config configs/mogo/v1_C3_mixB2_cam5lidar_volc_250418.yaml --checkpoint runs/v1_mix_0418/epoch_30.pth --rundir ./evaluate/v1_mix_0418 --eval bbox > v1_mix_0418_eval.txt 2>&1 

    # ============ 新增：动态查找最佳模型 ============
    # 查找 best_bbox_mAP 开头的模型文件
    BEST_MODEL=$(find $RUN_DIR -maxdepth 1 -name "best_bbox_mAP_*.pth" | head -n 1)
    
    if [ -z "$BEST_MODEL" ]; then
        # 回退机制：如果找不到最佳模型，使用最新的 lastest.pth
        echo "Warning: Best model not found! Using fallback: $MODEL"
    else
        MODEL=$BEST_MODEL
        echo "Using best model: $MODEL"
    fi

    SEND_MSG "${MSG1} using model: $MODEL"
    # =============================================

    torchrun --nproc_per_node=$GPUS \
     --master_port=$PORT \
     tools/test.py \
     --config $CONFIG \
     --checkpoint $MODEL \
     --rundir $EVAL_DIR \
     --launcher pytorch \
     --eval bbox > evaluate/v3_mixB2C1_0724_eval.txt 2>&1

    SEND_MSG "${MSG1} eval finish"
    sleep 30

else
    SEND_MSG "${MSG1} train failed"
fi

SEND_MSG "Occupying GPU \n${MSG1}"
python /pcpt/pcpt/project/jiangjunyuan/tools/gpu2.py --size 20000 --gpus 1 --interval 0.01

