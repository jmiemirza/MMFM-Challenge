# Overview
[MMFM Challenge](https://sites.google.com/view/2nd-mmfm-workshop/challenge?authuser=0).
Multimodal Foundation Models (MMFMs) have shown unprecedented performance in many computer vision tasks. However, on some very specific tasks like document understanding, their performance is still underwhelming. In order to evaluate and improve these strong multi-modal models for the task of document image understanding, we harness a large amount of publicly available and privately gathered data (listed in the image above) and propose a challenge. In the following, we list all the important details related to the challenge. Our challenge is running in two separate phases. For the first phase, we have released a training set consisting of publicly available data and for the second phase, an alien test set will be released soon.

This repository contains instructions on how to set up the code for the challenge which is part of our 
[The CVPR workshop on Multi-Modal Foundation Models](https://sites.google.com/view/2nd-mmfm-workshop/home?authuser=0) 
to appear in the [CVPR 2024](https://cvpr.thecvf.com/) program. 


For detailed instructions on how to setup the codebase and all the environments, and setup the code for evaluation, please read through these separate instructions:


## Table of Contents
1. [Getting Started](#instructions)
   1. [Installation](docs/installation.md)
   2. [Data Download](docs/download.md)
   3. [Evaluation](docs/evaluation.md)
   4. [Training](docs/training.md)
2. [Prize Money](#prize-money)
3. [Important Dates](#important-dates)
4. [Rules](#rules)
   - [General Rules](#general-rules)
   - [Challenge Phases](#challenge-phases)
   - [Metric](#metric)
   - [Open Source Models](#open-source-models)
5. [Submission](#submission)
6. [Baselines](#baselines)
7. [Disclaimer](#disclaimer)
8. [Contact](#contact)
9. [License](#license)


[//]: # (1. [Installation]&#40;docs/installation.md&#41;)

[//]: # (2. [Data Download]&#40;docs/download.md&#41;)

[//]: # (3. [Evaluation]&#40;docs/evaluation.md&#41;)

[//]: # (4. [Training]&#40;docs/training.md&#41;)

# Prize Money

<!-- The top 3 performers will be declared as the challenge winners and receive a prize totalling $10,000 USD, to be split as follows: -->

<!-- - 1st Place: $5000 -->
<!-- - 2nd Place: $3000 -->
<!-- - 3rd Place: $2000 -->


The top 3 performers will be declared as the challenge winners and receive a cash prize, details about this will be shared later.

The three winners will also be invited for spotlight talks at the workshop. Details about the talk will 
be shared with the winners after the challenge conclusion and before the workshop date.


# Important Dates
- Phase 1 deadline: 15th May 2024
- Phase 2 deadline: 30th May 2024
- Announcement of winners: 8th June 2024



# Rules

### General Rules
- To be eligible for participation in our challenge, you must register your team via CMT: [https://cmt3.research.microsoft.com/MMFM2024](https://cmt3.research.microsoft.com/MMFM2024).
    - Please select the `challenge track` while registering.

- The organizing committee reserves the right to verify the results and the code.
- The winners of the competition will be required to open source their code under MIT or more permissive licence. 
- For eligibility to win the prize money, the participants are required to open source their code and the model weights. For participants who submit results with closed-source models, the organizers will not be able to consider them for the prize money. This is to ensure that the organizers can verify the results and the code. All submitted results would be documented in a leaderboard, closed-source results will be marked.

### Challenge Phases
- The challenge will be running in two phases: 
  - Phase 1: The participants will submit their results on the test sets which is already present in `data/pre_processed` for all datasets.
  - Phase 2: An alien test set will be released after the phase 1 deadline. The participants will be required to submit their results on the alien test set. Again, with the code and the model weights (which should be same as phase 1). 
    - The alien test set will be of a similar domain to the current `test data`.
  - The model submitted for Phase 2 shall reproduce the results of Phase 1. This is stipulated in order to discourage people from overfitting on the test set. An immediate *disqualification* will be in place if the results of Phase 2 are not reproducible from the model submitted in Phase 1.
  - There is a 40% weightage for Phase 1 and 60% weightage for Phase 2.
  - Only participants that submitted results for Phase 1 before its deadline will receive access to Phase 2 evaluation data, and only participants that submitted Phase 2 results will be eligible for the competition.
  - Phase 2 is intended only for evaluating a model finalized in Phase 1, no model modification is allowed for Phase 2 (it should exactly reproduce Phase 1 evaluation).

- Important: To be eligible for the prize money, the participants are required to register, and participate in both phases of the challenge.


### Submission
- For all submission, the participants will be required to submit their code and the model weights with the instructions on how to reproduce the results. 
    - For Phase 1 The code should be submitted through a public `github repository` and the model weights should be uploaded to some storage. The teams are responsible for informing the organizers via email. 
      - Please email the organizers with following information. 
        - The `github repository` should contain:
          - The code for the submission
          - The results for the phases 1 on 10 datasets see below for details on the phases 
          - Details about the model architecture and the training details
          - The instructions on how to reproduce the results
          - The model weights (link to the storage where the weights are uploaded)
          - The `requirements.txt` file consisting the dependencies for the code
          - The code should be well-documented and easy to understand
- For Phase 2, please submit a 1-2 page report and submit the `github repository` link, and model weights link via [CMT]((https://cmt3.research.microsoft.com/MMFM2024)). 
  - The report should contain results for both pages and on overview of the methodology. 
    - The challenge winners will not be evaluated on the quality of the writing. 


### Metric
  The evaluation metric agreed upon by the organizers is: 

      Dataset Result = (performance of submitted model - performance of leading baseline) / (distance of leading baseline to 100%)
  the overall score will be computed as an average over the individual dataset results.

[//]: # (### Open Source Models)




# Baselines
Here we provide three baselines for the challenge by training the LLAVA-1.5 model on three types of data, and we provide the results obtained from the 
resulting model on the test sets of the 10 datasets which contain 200 randomly sampled from the original `validation split` of the dataset. 
Due to the nature of the datasets, we evaluate the models with two metrics (and report the Accuracy %): 
- The [MMMU metric](https://arxiv.org/abs/2311.16502). Used for 6 datasetes: `iconqa_fill, funsd, iconqa_choose, wildreceipt, textbookqa, tabfact`
- Using Mixtral as a judge to evaluate the outputs of the models. Used for 4 datasets: `docvqa, inforgraphicsvqa, websrc`


### Vanilla LLaVA Models:

- MMMU Evaluation 

| Model | Iconqa-Fill | Funsd | Iconqa-Choose | Wildreceipt | Textbookqa | Tabfact | Average |
|---------------|-------------|-------|---------------|-------------|------------|---------|---------|
| LLaVA 1.5 7B  | 13.5        | 21.5  | 36.0          | 6.0        | 37.5       | 54.0    | 28.1    |
| LLaVA 1.6 7B  | 13.0        | 17.5  | 38.5          | 20.0        | 44.0       | 49.0    | 30.3    |
| LLaVA 1.5 13B | 14.0        | 34.5  | 31.0          | 35.0        | 52.5       | 48.5    | 35.9    |
| LLaVA 1.6 13B | 14.5        | 39.5  | 35.0          | 44.5        | 54.5       | 47.5    | 39.3    |

- Mixtral Evaluation

| Model | DocVQA | InfographicsVQA | WebSRC | WTQ | Average |
|---------------|--------|-----------------|--------|-----|---------|
| LLaVA 1.5 7B  | 18.0   | 17.0            | 31.0   | 9.5 | 18.9    |
| LLaVA 1.6 7B  | 24.0   | 16.5            | 31.0   | 9.5 | 20.3    |
| LLaVA 1.5 13B | 22.5   | 20.5            | 29.5   | 8.0 | 20.1    |
| LLaVA 1.6 13B | 27.0   | 21.0            | 28.5   | 13.5 | 22.5    |



### LLaVA 1.5 13B instruction-tuned on the train sets of the 10 datasets:

- MMMU Evaluation 

| Iconqa-Fill | Funsd | Iconqa-Choose | Wildreceipt | Textbookqa | Tabfact | Average |
|-------------|-------|---------------|-------------|------------|---------|---------|
| 36.0        | 81.0  | 53.0          | 87.0        | 61.0       | 59.5    | 62.9    |

- Mixtral Evaluation

| DocVQA | InfographicsVQA | WebSRC | WTQ  | Average |
|--------|-----------------|--------|------|---------|
| 38.0   | 30.0            | 36.5   | 22.5 | 31.8    |

### LLaVA 1.5 13B instruction-tuned with LLAVA instruction-tuning data and the train sets of the 10 datasets:

- MMMU Evaluation 

| Iconqa-Fill | Funsd | Iconqa-Choose | Wildreceipt | Textbookqa | Tabfact | Average |
|-------------|-------|---------------|-------------|------------|---------|---------|
| 45.5        | 80.5  | 52.0          | 88.5        | 68.5       | 57.5    | 65.4    |

- Mixtral Evaluation

| DocVQA | InfographicsVQA | WebSRC | WTQ  | Average |
|--------|-----------------|--------|------|---------|
| 35.0   | 29.0            | 40.5   | 22.0 | 31.6    |

# Disclaimer
The organizers reserve the right to disqualify any participant who is found to be in violation of the rules of the challenge. 
The organizers also reserve the right to modify the rules of the challenge at any time.

# Contact
For any questions, please write an email to the organizers, a team member will get back to you as soon as possible:

[contactmmfm2024@gmail.com](contactmmfm2024@gmail.com).

# License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
To view the licenses of the datasets used in the challenge, please see [LICENSES](docs/licenses.md).
