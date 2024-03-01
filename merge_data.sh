#!/bin/bash

# use '.' by default
main_data_dir="${1:-.}"
python merge_datasets.py --max_samples 5000 --input_data_dir $main_data_dir/processed_data --save_dir $main_data_dir/merged_data
