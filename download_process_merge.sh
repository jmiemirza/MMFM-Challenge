#!/bin/bash

# use './data' by default
main_data_dir="${1:-./data}"

if [ ! -d "$main_data_dir" ]; then
  # The path does not exist, create it
  mkdir -p "$main_data_dir"
fi

bash download.sh $main_data_dir
bash process_data.sh $main_data_dir
bash merge_data.sh data/
