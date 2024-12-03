PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --main_process_port 29501 finetune.py \
	--model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
	--output_dir ./finetuned_7b \
	--cache_dir ~/.cache/ \
	--model_max_length 32768 \
	--use_flash_attn True \
	--low_rank_training True \
	--num_train_epochs 2 \
	--gradient_accumulation_steps 4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 4 \
	--save_strategy "steps" \
    --save_steps 500 \
	--learning_rate 1e-4 \
	--weight_decay 0.0 \
	--warmup_steps 30 \
	--save_total_limit 2 \
	--lr_scheduler_type cosine \
	--logging_steps 1 \
	--dataset_path "data/train_final.csv"

python3 prepare_model.py

for i in {1..7}
do
	CUDA_VISIBLE_DEVICES=$i python3 inference.py \
	--model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
	--output_dir ./thomas \
	--cache_dir ~/.cache/ \
	--rank $i \
	--file_name "response_test" \
	--model_max_length 32768 &
done

CUDA_VISIBLE_DEVICES=0 python3 inference.py \
	--model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
	--output_dir ./thomas \
	--cache_dir ~/.cache/ \
	--rank 0 \
	--file_name "response_test" \
	--model_max_length 32768
