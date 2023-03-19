#!/usr/bin/env bash

set -x

PARTITION=$1
NNum=1
GPUS_PER_NODE=1
PY_ARGS=${@:2}
JOB_NAME=eval
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    -N ${NNum} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u test.py ${PY_ARGS}
