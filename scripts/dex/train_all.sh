# © 2024 Nokia
# Licensed under the BSD 3-Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

#!/bin/bash

max_jobs_per_gpu=1
num_workers=4

seeds="0"
datasets="Imagenette" # Imagenette Caltech101 Caltech256 Food101
num_channels="64" #3 6 18 36 64 ; : coordconv: 5; coordconv(r): 6
models="simplenet" # simplenet widenet efficientnetv2 mobilenetv2_075
reshape_methods="--data-reshape" # "--coordconv" "--data-repeat" "--data-rotate" "--data-reshape"(dex)
with_r="" # "--with-r" hyperparam for coordconv
dex_configs="dex" # "dex" "tile_per_channel" "skewed_sample"

declare -a pid_array=()
declare -a cuda_array=(0 1 2 3 4 5 6 7) # Adjust if you have fewer devices
declare -a cuda_usage=(0 0 0 0 0 0 0 0) # Initialize with zero, tracking scripts per GPU


# Function to find and return the CUDA device with the least usage and less than 2 running tasks

find_available_cuda() {
  local min_usage=$max_jobs_per_gpu # Set initial minimal usage to max allowed jobs per GPU
  local min_index=-1 # Initialize with an invalid index

  for i in "${!cuda_usage[@]}"; do
    if [ "${cuda_usage[i]}" -lt "$max_jobs_per_gpu" ]; then
      if [ "${cuda_usage[i]}" -lt "$min_usage" ]; then
        min_usage="${cuda_usage[i]}"
        min_index=$i
      fi
    fi
  done

  if [ "$min_index" -ne -1 ]; then
    echo $min_index
    return 0 # Successfully found an available device
  else
    return 1 # Return an error if no available device is found
  fi
}

# Function to update CUDA usage tracking array
update_cuda_usage() {
  local device=$1
  local change=$2 # +1 or -1
  cuda_usage[$device]=$((${cuda_usage[$device]} + $change))
}
for seed in $seeds; do
  for dataset in $datasets; do

    for num_channel in $num_channels; do
      for model in $models; do
          for reshape in $reshape_methods; do
              for dex_config in $dex_configs; do

                common_args="--seed $seed --deterministic --workers $num_workers --validation-split 0 --dataset ${dataset}_${num_channel}x32x32 $reshape --data-reshape-method $dex_config $with_r" #--batch-size $batch_size

                if [ "$model" = "simplenet" ]; then # 32
                  args=$common_args" --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenet --batch-size 32 --device MAX78000 --print-freq 100 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"

                elif [ "$model" = "widenet" ]; then #100
                  args=$common_args" --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenetwide2x  --device MAX78000 --batch-size 100 --print-freq 100 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"

                elif [ "$model" = "efficientnetv2" ]; then #100
                  args=$common_args" --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100-effnet2.yaml --model ai87effnetv2 --device MAX78002 --batch-size 100 --print-freq 100 --use-bias --qat-policy policies/qat_policy_late_cifar.yaml"

                elif [ "$model" = "mobilenetv2_075" ]; then #128
                  args=$common_args" --epochs 300 --optimizer SGD --lr 0.1 --compress policies/schedule-cifar100-mobilenetv2.yaml --model ai87netmobilenetv2cifar100_m0_75 --batch-size 128 --device MAX78002 --print-freq 100 --use-bias --qat-policy policies/qat_policy_cifar100_mobilenetv2.yaml"
                fi

              # Wait for a CUDA device to be available if all are currently fully utilized
              while true; do
                cuda_id=$(find_available_cuda)
                if [ -n "$cuda_id" ]; then
                  update_cuda_usage $cuda_id 1
                  break
                fi
                # Periodically check and clean up finished jobs, updating GPU usage
                for idx in "${!pid_array[@]}"; do
                  if ! kill -0 ${pid_array[idx]} 2>/dev/null; then
                    update_cuda_usage $idx -1
                    unset pid_array[idx]
                  fi
                done
              done


              current_time=$(date "+%y%m%d-%H%M%S")
              echo "Executing on CUDA_VISIBLE_DEVICES=$cuda_id: python train.py $args"
              CUDA_VISIBLE_DEVICES=$cuda_id python train.py $args 2>&1 | tee logs/${current_time}_${dataset}_${model}_${num_channel}_${reshape:2}_${dex_config}_${seed}.txt &

              pid=$!
              pid_array[$cuda_id]=$pid
              sleep 1.5
            done
        done
      done
    done
  done
done

wait