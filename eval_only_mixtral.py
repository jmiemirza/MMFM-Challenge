"""Parse and Evalate"""
import json
import sys
import os
import os.path as osp
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'MMMU', 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'LLaVA'))

import pdb
from argparse import ArgumentParser
import numpy as np
import tqdm
import torch
import re
from torch import bfloat16

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.eval_utils import parse_open_response, calculate_ins_level_acc


def creat_prompt(model_id, tokenizer, question, answer, pred):
    messages = [
        {
            "role": "system",
            "content":
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
                "------\n"
                "##INSTRUCTIONS:\n"
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
        },
        {
            "role": "user",
            "content":
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question.capitalize()}\n"
                f"Correct Answer: {answer.lower()}\n"
                f"Predicted Answer: {pred.lower()}\n\n"
                "Evaluate if the answer is correct with yes/no and assign a correctness score between 0 and 5, where 0 indicates incorrect answer, and 5 signifies the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'no', 'score': 0}."
        }
    ]

    if 'mistralai' in model_id:
        prompt = f'<s>[INST] {messages[0]["content"].strip()}\n\n{messages[1]["content"].strip()} [/INST]'
    elif 'NousResearch' in model_id:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = prompt + '<|im_start|>assistant'
    else:
        raise NotImplementedError
    return prompt


def calculate_ins_level_score(results):
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results['avg_score'] * cat_results['num_example']
        ins_num += cat_results['num_example']
        return 0
    return acc / ins_num


def LLM_eval(model_id, model, tokenizer, batch_size, samples, cuda=True):

    steps = int(np.ceil(len(samples) / batch_size))
    evals = []
    for step in tqdm.tqdm(range(steps)):
        prompts = []
        for item in samples[step * batch_size: (step + 1) * batch_size]:
            question = item['question'].replace('<image>', '').strip()
            question = question.split('Context:')[0].strip()  # for ScienceQA
            answer = item['answer']
            pred = item['parsed_pred']
            prompt = creat_prompt(model_id, tokenizer, question, answer, pred)
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        if cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs = tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        evals.extend(outputs)
    judge_dict = dict()
    pred_correct = 0
    score_sum = 0

    for sample_data, sample_eval in zip(samples, evals):
        try:
            sample_eval = re.match(r".*(\{.*?\}).*", sample_eval, re.S).group(1)
            sample_eval = sample_eval.replace("'", '"')
            sample_eval = json.loads(sample_eval)
            pred = sample_eval['pred']
            sample_score = sample_eval['score']

            if pred == 'yes':
                judge_dict[sample_data['id']] = {'pred': 'Correct', 'score': sample_score}
                pred_correct += 1
            else:
                judge_dict[sample_data['id']] = {'pred': 'Wrong', 'score': sample_score}
            score_sum += sample_score

        except:
            judge_dict[sample_data['id']] = {'pred': 'Wrong', 'score': 0}

    if len(samples) == 0:
        return {'acc': 0, 'avg_score': 0}
    return judge_dict, {'acc': pred_correct / len(samples), 'avg_score': score_sum / len(samples)}

def get_category_name(data_id):
    items = data_id.split("_")[0:2]
    if items[0] == items[1]:
        return items[0]
    else:
        return '_'.join(items)
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./example_outputs/qwen_vl/total_val_output.json", help="The path to model output file.")
    parser.add_argument('--answer_path', type=str, default="./answer_dict_val.json", help="Answer file path.")
    parser.add_argument('--llm_eval_score_save_dir', type=str, default="llm_eval_score_save_dir")
    parser.add_argument('--debug', action='store_true', help='enable debugger')
    parser.add_argument('--model_id', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cat', type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if 'mistralai' in args.model_id:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                 device_map='cpu' if args.cpu else 'auto',
                                                 torch_dtype=bfloat16,
                                                 load_in_8bit=args.load_in_8bit,
                                                 load_in_4bit=args.load_in_4bit)

    output_dict = json.load(open(args.output_path))
    answer_dict = json.load(open(args.answer_path))

    # group by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        # category = data_id.split("_")[0] #[1:-1]
        category = get_category_name(data_id)
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({ '_'.join(data_id.split('_')[:-1]) : parsed_pred})

    # group by category
    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        # category = data_id.split("_")[0] #[1:-1]
        category = get_category_name(data_id)
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({ '_'.join(data_id.split('_')[:-1]) : parsed_pred})

    evaluation_result = {}

    all_cats = list(answer_dict_w_cat.keys())
    if args.cat is not None:
        all_cats = [args.cat]

    DOMAIN_CAT2SUB_CAT = {'doc-vl': all_cats}

    for category in all_cats: # CAT_SHORT2LONG.values():
        print("Evaluating: {}".format(category))
        # get cat_outputs and cat_answers
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            print("Skipping {} for not found".format(category))
            continue
        
        exampels_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            question_type = cat_answers[data_id]['question_type']
            exampels_to_eval.append({
                "id": data_id,
                "question_type": question_type,
                "answer": cat_answers[data_id]['ground_truth'],
                "question": cat_answers[data_id]['question'],
                "parsed_pred": parsed_pred
            })

        if args.debug:
            exampels_to_eval = exampels_to_eval[:128]

        judge_dict, metric_dict = LLM_eval(args.model_id, model, tokenizer, args.batch_size, exampels_to_eval, cuda=not(args.cpu))
        metric_dict.update({"num_example": len(exampels_to_eval)})

        evaluation_result[category] = metric_dict

        judge_dict_path = f'judge_dict_{category}.json'
        judge_dict_path = judge_dict_path.replace('/', '_')
        if not os.path.exists(args.llm_eval_score_save_dir):
            os.makedirs(args.llm_eval_score_save_dir)
        with open(os.path.join(args.llm_eval_score_save_dir, judge_dict_path), 'w') as fout:
            json.dump(judge_dict, fout)

    printable_results = {}

    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats: # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_ins_score = calculate_ins_level_score(in_domain_cat_results)
        in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
        printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                  "acc": round(in_domain_ins_acc, 3),
                                                  "avg_score": round(in_domain_ins_score, 3),
                                                  }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                           "acc": round(cat_results['acc'], 3),
                                           "avg_score": round(cat_results['avg_score'], 3)
                                           }
        
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    all_ins_score = calculate_ins_level_score(evaluation_result)
    printable_results['Overall'] = {"num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
                                    "acc": round(all_ins_acc, 3),
                                    "avg_score": round(all_ins_score, 3)
                                    }

    # print(printable_results)
    print('\n')
    for key, value in printable_results.items():
        print(f'{key:<30} {value["num"]:<10} {value["acc"]:<10}')

