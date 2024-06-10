# Multimodal Structured Generation for Cheap & SOTA Document-Image Understanding

Team Name: `leloy`

Members: [Franz Louis Cesista](https://huggingface.co/leloy)

GitHub Repo: https://github.com/leloykun/MMFM-Challenge

My codes are in the [mmfm](https://github.com/leloykun/MMFM-Challenge/tree/mmfm) branch. This repo contains jupyter notebooks, scripts for running inference on the cloud (Huggingface Inference Endpoints & Modal Labs), and all inference results I've produced during the challenge.

---

## (Preliminary) Main Results

1. **Structured Generation can supplant finetuning, and maybe even multimodality, for document understanding.** The language-only foundation model (NousHermes 2 Pro) I augmented with Structured Generation outperformed finetuned multimodal models on the MyDoc dataset. Likewise, the vanila Llava-Next model augmented with Structured Generation outperformed even finetuned multimodal models I tried on the MyInfographic dataset. This shows the effectiveness of Structured Generation and should be the first thing teams should try when tackling document-image understanding tasks.

This is perhaps the first in-the-wild experimental support of the results in our paper, [Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Use](https://arxiv.org/abs/2405.20245v1).

2. **Vanila open-source models are still bad at Chart Understanding** All of the open-source models I tried (multimodal, finetuned, augmented with structured generation, etc.) failed at the MyChart dataset.

---

. | **MyDoc** | **MyChar** | **MyInfographic** | **Overall Acc**
--- | --- | --- | --- | ---
My Solution | 62.25\% | 4.5\% | 60.98\% | 50.49\%

---

## How To Reproduce

### Phase 1

I didn't focus too much on Phase 1 as the test set was already released from the outset and therefore not a true test set anymore. I propose we exclude this from the final evaluation.

That said, here's how you can reproduce my Phase 1 results with just 1 line of code:

```bash
MODEL_ID=liuhaotian/llava-v1.5-13b \
  modal run modal_llava_inference.py::run_inference_llava \
  --datasets="docvqa,infographicvqa,websrc,wtq,iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact" \
  --output-file-name="output_datasets_all--liuhaotian/llava-v1.5-13b.json"
```

The first run should take ~3 hours (due to the dataset preparation step) and cost ~\$5 or so on Modal Labs. But subsequent runs should only take ~30 mins and cost ~\$1.

You can then download the results file from Modal using `modal volume get mmfm-inference-results <results_file>.json inference_results/`.

### Phase 2

Upon manual inspection and simple color analysis (see `1b-analysis-document-colorfulness.ipynb`), I noticed that the documents MyChart and MyInfographic are a lot _less_ dense than the ones in MyDoc. Thus, I figured that a typical visual-language model should suffice for these datasets. To reproduce my results, you can follow these steps:

1. First, you need to create a Huggingface Inference Endpoint for the Llava-Next model. You can do this by running the `0c-create-huggingface-endpoint-llava-1.6.ipynb` notebook.
2. Then, run the `3b-mychart-llava-1.6.ipynb` and `3c-myinfographic-llava-1.6.ipynb` notebooks. Note that you might need to re-run them with different seeds in case some of the runs fail.

The documents in the MyDoc dataset, however, are a lot denser. They could contain thousand of text tokens, yet even Llava-Next only uses hundreds of visual tokens. Thus, I figured that a properly-augmented LLM should be able to outperform multimodal models. And I was right. To reproduce my results, you can follow these steps:

1. First, run OCR on the MyDoc dataset using Surya by running `modal run modal_surya.py::run_surya_extraction --dataset="mydoc"`.
2. Then, create the Huggingface Inference Endpoint by runnign the `0d-create-huggingface-endpoint-nous-hermes-docile.ipynb` notebook.
3. Finally, run the `5b-mydoc-nous-hermes-docile.ipynb` notebook. Note that you might need to re-run them with different seeds in case some of the runs fail.

### Formatting the solution for submission

To generate the final solution you can submit to the challenge platform, simply run the `build-submission.ipynb` notebook.

---

Models Used:
- [Llava v1.5 13B](https://huggingface.co/liuhaotian/llava-v1.5-13b) for all of Phase 1
- [Llava-Next](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) for MyChart & MyInfographic datasets in Phase 2
- [Surya](https://github.com/VikParuchuri/surya) for document OCR
- [(Finetuned) Nous Hermes 2 Pro](https://huggingface.co/leloy/Nous-Hermes-2-Pro-Docile-RASG-1ShotRetrieval-StructuredPrompt) for the MyDoc dataset in Phase 2. This model was finetuned on the [DocILE Dataset](https://github.com/rossumai/docile) which is also open-source.

Other technologies used:
- [Modal Labs](https://modal.com/) for running jobs on the cloud
- [Huggingface Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated) for setting up inference endpoints for the open-source models I used
- [Huggingface Text Generation Interface](https://huggingface.co/docs/text-generation-inference/en/index) cuz it makes it a lot simpler to combine multimodality with structured generation.

Open-source models tried, but not used for the final solution:
- [LLava v1.5 finetuned on MMMU for 10 epochs](https://huggingface.co/zhiqiulin/llava-v1.5-7b-MMMU-epoch-10)
- [PaliGemma](https://huggingface.co/google/paligemma-3b-mix-448)
- [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b)
- [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa)
- [Donut Chart2Table](https://huggingface.co/khhuang/chart-to-table)

Closed-source models tried, but not used for the final solution:
- [GPT-4o](https://openai.com/index/hello-gpt-4o/)
