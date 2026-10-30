import os
import subprocess

# 设置环境变量
os.environ['PYTHONPATH'] = '/rbs/szh/Projects/bevfusion/multimodal_bevfusion'

# 构建命令
command = [
    "python", "-u", "-m", "torch.distributed.launch",
    "--nproc_per_node=4",
    "--nnodes=1",
    "--node_rank=0",
    "--master_addr=10.0.32.60",
    "--master_port=1234",
    "tools/train.py",
    "--config", "configs/mogo/at128_bigfusionpkl_lidaronly.yaml",
    "--launcher", "pytorch",
    "--run_dir", "output_lw3"
]

# 运行命令
subprocess.run(command)
