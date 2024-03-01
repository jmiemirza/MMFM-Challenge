#!/bin/bash
#API_KEY=$1

# ===== KIE =====
# use '.' by default
main_data_dir="${1:-.}"


echo "Processing websrc"
python data_preprocessors/websrc.py --input_data_dir $main_data_dir/raw_datasets/websrc/release --main_data_dir $main_data_dir --out_data_dir data/processed_data/websrc

echo "Processing docvqa"
python data_preprocessors/docvqa.py --input_data_dir $main_data_dir/raw_datasets/docvqa/aws_neurips_time/docvqa --main_data_dir $main_data_dir --out_data_dir data/processed_data/docvqa --img_data_dir $main_data_dir/raw_datasets/docvqa/jpgs

echo "Processing funsd"
python data_preprocessors/funsd.py --input_data_dir $main_data_dir/raw_datasets/funsd/dataset --main_data_dir $main_data_dir --out_data_dir data/processed_data/funsd

echo "Processing iconqa"
python data_preprocessors/iconqa.py --input_data_dir $main_data_dir/raw_datasets/iconqa/iconqa_data/iconqa/ --main_data_dir $main_data_dir --out_data_dir data/processed_data/iconqa

echo "Processing infographicvqa"
python data_preprocessors/infographicvqa.py --input_data_dir $main_data_dir/raw_datasets/infovqa/aws_neurips_time/infographics_vqa --main_data_dir $main_data_dir --out_data_dir data/processed_data/infographicvqa --img_data_dir $main_data_dir/raw_datasets/infovqa/jpgs

echo "Processing tabfact"
python data_preprocessors/tabfact.py --input_data_dir $main_data_dir/raw_datasets/tabfact/aws_neurips_time/TabFact --main_data_dir $main_data_dir --out_data_dir data/processed_data/tabfact --img_data_dir $main_data_dir/raw_datasets/tabfact/jpgs

echo "Processing textbookqa"
python data_preprocessors/textbookqa.py --input_data_dir $main_data_dir/raw_datasets/textbookqa/tqa_train_val_test --main_data_dir $main_data_dir --out_data_dir data/processed_data/textbookqa

echo "Processing wildreceipt"
python data_preprocessors/wildreceipt.py --input_data_dir $main_data_dir/raw_datasets/wildreceipt/wildreceipt --main_data_dir $main_data_dir --out_data_dir data/processed_data/wildreceipt

echo "Processing wtq"
python data_preprocessors/wtq.py --input_data_dir $main_data_dir/raw_datasets/wtq/aws_neurips_time/WikiTableQuestions --main_data_dir $main_data_dir --out_data_dir data/processed_data/wtq --img_data_dir $main_data_dir/raw_datasets/wtq/jpgs/

