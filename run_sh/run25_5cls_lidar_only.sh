torchpack dist-run -np 8 python tools/train.py configs/mogo/falcon_v25_lidaronly_e10.yaml --run-dir runs/falcon_v25_lidaronly_e10 > falcon_v25_lidaronly_e10.txt 2>&1
# torchpack dist-run -np 8 python tools/train.py configs/mogo/falcon_debug.yaml 
# torchpack dist-run -np 8 python tools/train.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_lidar_only_noaug_use4dim.yaml --resume_from runs/falcon_7000_5class_lidar_only_noaug_use4dim/epoch_1.pth --run-dir runs/falcon_7000_5class_lidar_only_noaug_use4dim > falcon_7000_5class_lidar_only_noaug_use4dim.txt 2>&1
# torchpack dist-run -np 8 python tools/train.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_noaug.yaml --run-dir runs/falcon_7000_5class_noaug > falcon_7000_5class_noaug.txt 2>&1
# torchpack dist-run -np 8 python tools/train.py configs/mogo/falcon_7000_5class_fixCBGS_fixheatmap_noaug_use4dim.yaml --run-dir runs/falcon_7000_5class_noaug_use4dim > falcon_7000_5class_noaug_use4dim.txt 2>&1
python /rbs/houjiayue/gpu2.py --gpus 1 --size 20000 --interval 0.01

