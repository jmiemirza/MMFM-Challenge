
# Data Download & Process 
## Phase 1
The datasets will be downloaded into `data/raw_datasets` in the current repo. The datasets take around 90G. The dataset-wise train/val/test splits and merged splits will be saved in `data/processed_data` and `data/merged_data`
The test splits are already provided in `data/pre_processed` for all datasets.
The following datasets have been chosen for the challenge:

1. [DocVQA](https://www.docvqa.org/datasets)
2. [FUNSD](https://guillaumejaume.github.io/FUNSD/)
3. [IconQA](https://iconqa.github.io/)
4. [InfogrpahicVQA](https://www.docvqa.org/datasets/infographicvqa)
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
## Phase 2

The data for the phase 2 can be downloaded from this [Google Drive Link](https://drive.google.com/file/d/1Nnh_5LN6wf_byJvINzf5CXRoLIj6DbEH/view).

There are 3 datasets in total.  
- mydoc: 360 document images, 400 questions (w/o answer)
- mychart: 200 chart images, 200 questions (w/o answer)
- myinfographic: 428 infographic images, 428 questions (w/o answer)

The folder structure is as follows:
```
- mydoc
  - images
    - xx.png
    - ...
  - annot_wo_answer.json
- mychart
    - images
        - xx.png
        - ...
    - annot_wo_answer.json
- myinfographic
    - images
        - xx.jpg
        - ...
    - annot_wo_answer.json
```
The annotation is in the same format as the phase 1 data except that the answer is not provided. 
```
    {
        "id": "myinfographic_5",
        "image": "bally-chohan-recipe--healthy-herbs-for-healing_53b18ea91da93_w450_h300.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nCan you list at least three food pairings for basil as per the infographic?"
            }
        ]
    }
```

To obtain the results for participating in the challenge, the output files have to be evaluated through our evaluation server. Please see the [Submission](https://github.com/jmiemirza/MMFM-Challenge?tab=readme-ov-file#submission) subsection for details. 
