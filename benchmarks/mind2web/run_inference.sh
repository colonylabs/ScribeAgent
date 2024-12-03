for i in {1..7}
do
	CUDA_VISIBLE_DEVICES=$i python3 inference.py \
	--model_name_or_path "Qwen/Qwen2-7B-Instruct" \
	--output_dir ./7b_vllm \
	--cache_dir ~/.cache/ \
	--rank $i \
	--file_name "test_task_rope_4_response" \
	--model_max_length 130900 \
	--test_file_path "test_task_rope_4.csv" \
	--rope_scaling 4 &
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 inference.py \
	--model_name_or_path "Qwen/Qwen2-7B-Instruct" \
	--output_dir ./7b_vllm\
	--cache_dir ~/.cache/ \
	--file_name "test_task_rope_4_response" \
	--model_max_length 130900 \
	--test_file_path "test_task_rope_4.csv" \
	--rope_scaling 4 \
