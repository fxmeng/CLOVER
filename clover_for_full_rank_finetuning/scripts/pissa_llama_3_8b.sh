BASE_MODEL="meta-llama/Meta-Llama-3-8B"
RES_MODEL="output/PiSSA-Llama-3-8b-r32"
OUTPUT_PATH="output/commonsense-PiSSA-Llama-3-8b-r32"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com
if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform PiSSA initialization by my self."
    python utils/init_pissa.py --base_model_path $BASE_MODEL --output_dir $RES_MODEL --init_weights pissa_niter_16 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --target_modules q_proj k_proj v_proj up_proj down_proj
fi
# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $RES_MODEL \
    --full_finetune False \
    --bf16 \
    --init_weights pissa \
    --adapter_name_or_path "pissa_init" \
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
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type "linear" \
    --report_to "tensorboard" \
    --merge True \

python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task commonsense --output_file $OUTPUT_PATH/all_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/all_response.jsonl 