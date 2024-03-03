
## Inference on the test model
#### Run the inference on LLaVA 1.5
Activate the LLaVA environment
```
conda activate MMFM-Challenge
```
Run the inference on the test data of the 4 datasets of open-ended questions: `docvqa,infographicvqa,websrc,wtq`. This generates the output file of predicted answers `results/output_4_datasets.json`. 
The predicted answers on these 4 datasets will be evaluated by the LLM - Mixtral 8x7b model. Specify the path to the processed data splits `data/processed_data` and the path to the LLaVA 1.5 model `/llava/model/path`
```
cd ../MMFM-Challenge
CUDA_VISIBLE_DEVICES=0 python eval_llava.py \
  --output_path inference_results/output_4_datasets.json \
  --data_path data/processed_data \  # path to the processed data splits
  --model_path /llava/model/path \ # path to the LLaVA 1.5 model
  --test_file_name converted_output_test.json \
  --sub_ds_list docvqa,infographicvqa,websrc,wtq
```
Run the inference on the test data of the 6 datasets: `iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact`. This generates the output file of predicted answers `results/output_6_datasets.json`.
The predicted answers on these 6 datasets will be evaluated with the MMMU metric.

```
# run the inference on the 6 datasets: iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact
CUDA_VISIBLE_DEVICES=0 python eval_llava.py \
  --output_path inference_results/output_6_datasets.json \
  --data_path data/processed_data \  # path to the processed data splits
  --model_path /llava/model/path \ # path to the LLaVA 1.5 model
  --test_file_name converted_output_test.json \
  --sub_ds_list iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact
```

## Evaluation
#### Evaluation with MMMU metric on the 6 datasets: 
Run the MMMU evluation on 6 datasets: `iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact`.

First build ground truth answer dictionary for all datasets. This generates the ground truth answer file `answer_dicts/coverted_output_test_all_datasets.json`
```
python build_answer_dict_val.py \
--output_path answer_dicts/coverted_output_test_all_datasets.json \
--test_file_name converted_output_test.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq,iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact \
--data_path data/processed_data
```
Run the MMMU evluation on 6 datasets. feeding the output file of predicted answers `inference_results/output_6_datasets.json` generated in the previous step, and the ground truth answer file `answer_dicts/coverted_output_test_all_datasets.json`
```
python eval_only.py --output_path inference_results/output_6_datasets.json --answer_path answer_dicts/coverted_output_test_all_datasets.json
```



#### LLM (Mixtral 8x7b)-based evaluation on 4 datasets with open-ended questions
Run the LLM (Mixtral 8x7b)-based evaluation for the predicted answers in the 4 datasets of open-ended questions: `docvqa,infographicvqa,websrc,wtq`.
Prepare the QA pairs for the 4 datasets of open-ended questions. This creates the directory `data/processed_data_for_mixtral_eval` abd store the QA pairs (without instruction templates) there.
```
bash process_data_for_mixtral_eval.sh
```

Then build ground truth QA dictionary for LLM-based evaluation. This generates the ground truth QA dictionary file `converted_output_val_for_mixtral_eval.json`
```
python build_answer_dict_val_w_question.py \
--output_path answer_dicts/converted_output_test_for_mixtral_eval.json \
--test_file_name converted_output_test.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq \
--data_path data/processed_data_for_mixtral_eval
```
Run the Mixtral evaluation (in 4-bit for fast prompting) on 4 datasets, feeding the output file of predicted answers `results/output_4_datasets.json` generated in the previous step, and the ground truth QA dictionary file `converted_output_val_for_mixtral_eval.json`
This outputs the evaluation results for each dataset, and writes the evaluation on all samples into `llm_eval_score_save_dir/judge_dict_{dataset}.json` files


Activate the Mixtral environment
```
conda activate mixtral
```
Run the Mixtral evaluation on 4 datasets
```
CUDA_VISIBLE_DEVICES=0 python eval_only_mixtral.py \
--output_path inference_results/output_4_datasets.json \  # feeding the output file `results/output_4_datasets.json` generated in the previous step
--llm_eval_score_save_dir llm_eval_score_save_dir \  # directory to save the llm evaluation scores
--answer_path answer_dicts/converted_output_test_for_mixtral_eval.json \  # QA dictionary file 
--model_id mistralai/Mixtral-8x7B-Instruct-v0.1 \ # here model_id is the model id of the Mixtral 8x7b model or the path to the model
--load_in_4bit
```