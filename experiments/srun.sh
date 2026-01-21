#!/bin/bash

#export NCCL_DEBUG=info
#export NCCL_P2P_DISABLE=1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
num_of_frame=4
btz=32
ds_factor=8
ds_factor_low=32
img_res=256
output_dir=output

torchrun --nnodes=1 --nproc_per_node=4 --master_port=${port} \
  main.py --img_size 256 --batch_size $btz --resume $output_dir \
    --num_of_frame $num_of_frame --output_dir $output_dir \
    --ds_factor $ds_factor --ds_factor_low $ds_factor_low \