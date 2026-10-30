#!/usr/bin/env bash

set -x
CONFIG=$1
NGPUS=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
PY_ARGS=${@:6}

python -u -m torch.distributed.launch \
            --nproc_per_node=${NGPUS} \
            --nnodes=${NNODES} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=1234 \
            tools/train.py \
            --config $CONFIG \
            --launcher pytorch \
            ${PY_ARGS}

