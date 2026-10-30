# bash tools/dist_train.sh\
#     configs/mogo/v23_cutPasteNoise.yaml\
#     4 1 0 10.0.33.157 --run-dir ../runs/v23_cutPasteNoise > ../log/v23_cutPasteNoise.txt 2>&1
# python /rbs/houjiayue/gpu2.py --gpus 1 --size 20000 --interval 0.01

python tools/train.py --config configs/mogo/v2_C3_mixB2_llm45k_cam5lidar_250604.yaml --run-dir runs/v2_mix_0612_e50 > log/v2_mix_0612_e50_test.txt 2>&1
python /rbs/houjiayue/gpu2.py --gpus 1 --size 20000 --interval 0.01