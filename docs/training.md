## Train a model on the train portions of the challenge data
#### Run the inference on LLaVA 1.5
Activate the LLaVA environment
```
conda activate MMFM-Challenge
```
Download the LLaVA instruction tuning data following the instructions in the [LLaVA repo](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning). 
Place the json file `llava_v1_5_mix665k.json` in `LLaVA/data`. 
Organize the image data in the directory of `./data/processed_data`.

Download the pretrained projector weights of LLaVA 1.5 from the [Huggingface repo](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5) into the directory of `LLaVA/checkpoints`

Build the visual instruction tuning data mix. Use `--mixin` parameter to mix additional instruction tuning data (e.g. LLaVA finetuning JSON):
```
python build_train_data.py \
--output_path ./data/data_mix.json \
--train_file_name converted_output_train.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq,iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact \
--data_path data/processed_data \
--mixin lib/LLaVA/data/llava_v1_5_mix665k.json
```
Launch LLaVA 1.5 training, first argument is the name of the data mix JSON, the second is the root folder for the images (JSON provides all images as relative paths under this root):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 source scripts/finetune_vicuna13b_clm.sh data_mix ./data/processed_data
```
