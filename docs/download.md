
# Data Download Instructions 
The datasets will be downloaded into `data/raw_datasets` in the current repo. The datasets take around 90G. The dataset-wise train/val/test splits and merged splits will be saved in `data/processed_data` and `data/merged_data`
The test splits are already provided in `data/pre_processed` for all datasets.
The following datasets have been chosen for the challenge:

1. [DocVQA](https://www.docvqa.org/datasets)
2. [FUNSD](https://guillaumejaume.github.io/FUNSD/)
3. [IconQA](https://iconqa.github.io/)
4. [InfogrpahicsVQA](https://www.docvqa.org/datasets/infographicvqa)
5. [Tabfact](https://tabfact.github.io/) 
6. [TextbookVQA](https://allenai.org/data/tqa)
7. [WebSrc](https://x-lance.github.io/WebSRC/)
8. [Wildreceipt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/dataset/kie_datasets_en.md#wildreceipt-dataset) 
9. [WTQ](https://github.com/ppasupat/WikiTableQuestions)


For downloading, processing and merging all these datasets, please run: 
```bash
conda activate MMFM-Challenge
bash download_process_merge.sh
```