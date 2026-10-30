# torchpack dist-run -np 8 python tools/test.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e6.yaml ../runs/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e6/epoch_6.pth --eval bbox > falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e6_eval2.txt 2>&1 
# torchpack dist-run -np 8 python tools/test.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e10.yaml ../runs/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e10/epoch_10.pth --eval bbox > falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e10_eval2.txt 2>&1 
# torchpack dist-run -np 8 python tools/test.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e20.yaml ../runs/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e20/epoch_20.pth --eval bbox > falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e20_eval2.txt 2>&1 
# torchpack dist-run -np 8 python tools/test.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e40.yaml ../runs/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e40/epoch_40.pth --eval bbox > falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim_origin05_bevpos_perbike_e40_eval2.txt 2>&1 
# ps -ef|grep python |grep -v grep|awk '{print $2}'|xargs kill -9
# torchpack dist-run -np 4 python tools/test.py configs/mogo/at128_MM_proposal200_max_voxels_10cm_0529pkl_rot45_minPts10_lidaronly_upsampleSegLossWeight2_lm.yaml ../runs/at128_MM_proposal200_max_voxels_10cm_0529pkl_rot45_minPts10_lidaronly_upsampleSegLossWeight2_lm/epoch_30.pth --eval bbox > at128_MM_proposal200_max_voxels_10cm_0529pkl_rot45_minPts10_lidaronly_upsampleSegLossWeight2_lm_eval2_fixcoder.txt 2>&1 
# torchpack dist-run -np 4 python tools/test.py --config configs/mogo/v1_C3_cam5lidar.yaml --checkpoint ./runs/v1_c3/epoch_10.pth --eval bbox > v1_C3_cam5lidar.txt 2>&1 
# python tools/test.py --config configs/mogo/v1_C3_mixB2_cam5lidar.yaml --checkpoint runs/v1_c3_mix/epoch_30.pth --esval bbox 
# python /rbs/houjiayue/gpu2.py --gpus 1 --size 20000 --interval 0.01

MSG1="v3_mixB2C1_0724"
GPUS=8
PORT=11985
CONFIG=configs/sweeper/v3_C1_mixB2_45kllm35k_cam7lidar_voxelpoint20_250724.yaml
RUN_DIR=runs/v3_mixB2C1_0724
EVAL_DIR=evaluate/v3_mixB2C1_0724_e30
MODEL=$RUN_DIR/epoch_30.pth

torchrun --nproc_per_node=$GPUS \
     --master_port=$PORT \
     tools/test.py \
     --config $CONFIG \
     --checkpoint $MODEL \
     --rundir $EVAL_DIR \
     --launcher pytorch \
     --eval bbox > evaluate/v3_mixB2C1_0724_e30.txt 2>&1

python /pcpt/pcpt/project/jiangjunyuan/tools/gpu2.py --size 20000 --gpus 1 --interval 0.01