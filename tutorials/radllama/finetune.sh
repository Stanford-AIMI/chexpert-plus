#!/bin/bash

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=${SLURM_NNODES}
NODE_RANK=${SLURM_NODEID}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL="StanfordAIMI/RadLLaMA-7b"
DATA="<PATH TO YOUR DATA>"
output_dir=outputs

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path "$DATA" \
    --fp16 False \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --group_by_length True \
    --dataloader_num_workers 4 \
    --deepspeed ds_config_zero3.json
