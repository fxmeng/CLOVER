BASE_MODEL="/data2/mengfanxu/huggingface/Meta-Llama-3-8B"
OUTPUT_PATH="output/commonsense-CLOVER-Llama-3-8b"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16974 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune True \
    --init_weights clover \
    --bf16 \
    --data_path $DATA_PATH \
    --sub_task commonsense \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type "linear" \
    --report_to "tensorboard" \
    --merge True \
    --shuffle_dataset True

python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task commonsense --output_file $OUTPUT_PATH/all_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/all_response.jsonl 