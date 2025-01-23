#!/bin/sh
RES_MODEL="output/CLOVER-Llama-2-7b-64-qkvud"
OUTPUT_PATH="output/commonsense-CLOVER-Llama-2-7b-64-qkvud"
lr=1e-4
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000
do 
echo $i
python pissa-dataset/merge_adapter.py --base_model $RES_MODEL --adapter $OUTPUT_PATH/$lr/checkpoint-$i/adapter_model/ --output_path $OUTPUT_PATH/$lr/checkpoint-$i/merged_model &&
python gen_vllm.py --model $OUTPUT_PATH/$lr/checkpoint-$i/merged_model --sub_task boolq piqa siqa hellaswag winogrande arc_easy arc_challenge openbookqa --output_file $OUTPUT_PATH/$lr/checkpoint-$i/all_response.jsonl &&
rm -rf $OUTPUT_PATH/$lr/checkpoint-$i/merged_model
done
python test_acc.py --input_file $OUTPUT_PATH/$lr/checkpoint-$i/all_response.jsonl 