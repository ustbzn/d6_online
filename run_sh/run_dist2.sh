# ps -ef|grep python |grep -v grep|awk '{print $2}'|xargs kill -9
bash tools/dist_train.sh\
    configs/mogo/falcon_v4data2CPureclsv4Ptsnum_5class_fixCBGS_fixheatmap_lidar_only_aug_use4dim_origin05_bevpos_perbike_yawcls_lossbox1_fixres2_e30_dist.yaml\
    1 2 1 10.0.33.156


# python /rbs/houjiayue/gpu2.py --gpus 1 --size 20000 --interval 0.01

