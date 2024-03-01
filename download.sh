#!/bin/bash

#main_data_dir=/system/user/publicdata/LMM_benchmarks/MMFM_challenge_test_pipeline/

# use '.' by default
main_data_dir="${1:-.}"

data_dir=$main_data_dir/raw_datasets
export DATASET_DIR=$data_dir
mkdir -p $data_dir

sh ./download_scripts/due.sh
sh ./download_scripts/websrc.sh
sh ./download_scripts/funsd.sh
sh ./download_scripts/iconqa.sh
sh ./download_scripts/textbookqa.sh
sh ./download_scripts/screen2words.sh
sh ./download_scripts/wildreceipt.sh


# font file for rendering text in AI2D dataset
wget https://huggingface.co/Team-PIXEL/pixel-base-finetuned-masakhaner-swa/resolve/main/GoNotoCurrent.ttf
