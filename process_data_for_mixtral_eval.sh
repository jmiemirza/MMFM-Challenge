#!/bin/bash
#API_KEY=$1

# ===== KIE =====
# use '.' by default
main_data_dir="${1:-./data}"

echo "Processing docvqa"
python data_preprocessors/docvqa_for_llm_eval.py \
--input_data_dir $main_data_dir/raw_datasets/docvqa/aws_neurips_time/docvqa \
--main_data_dir $main_data_dir \
--out_data_dir data/processed_data/docvqa \
--img_data_dir $main_data_dir/raw_datasets/docvqa/jpgs \
--remove_instruct_templates

echo "Processing websrc"
python data_preprocessors/websrc_for_llm_eval.py \
--input_data_dir $main_data_dir/raw_datasets/websrc/release \
--main_data_dir $main_data_dir \
--out_data_dir data/processed_data/websrc \
--remove_instruct_templates

echo "Processing infographicvqa"
python data_preprocessors/infographicvqa_for_llm_eval.py \
--input_data_dir $main_data_dir/raw_datasets/infovqa/aws_neurips_time/infographics_vqa \
--main_data_dir $main_data_dir \
--out_data_dir data/processed_data/infographicvqa \
--img_data_dir $main_data_dir/raw_datasets/infovqa/jpgs \
--remove_instruct_templates


echo "Processing wtq"
python data_preprocessors/wtq_for_llm_eval.py \
--input_data_dir $main_data_dir/raw_datasets/wtq/aws_neurips_time/WikiTableQuestions \
--main_data_dir $main_data_dir \
--out_data_dir data/processed_data/wtq \
--img_data_dir $main_data_dir/raw_datasets/wtq/jpgs \
--remove_instruct_templates