import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'MMMU', 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'LLaVA'))

import torch
import random

import hashlib

import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser

from utils.data_utils import load_yaml, save_json

# sub_ds_list = ['llavar', 'docvqa', 'infographicsvqa', 'chartqa/val', 'scienceqa/val']
# test_file_name = 'converted_output_val.json'

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='answer_dict_val.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="lib/MMMU/eval/configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="/dccstor/leonidka1/irenespace/mmfm_challenge/data") # hf dataset path: MMMU/MMMU
    parser.add_argument('--split', type=str, default='validation')

    parser.add_argument('--test_file_name', type=str, default='converted_output_val.json')
    parser.add_argument('--sub_ds_list', type=str, default='llavar,docvqa,infographicsvqa,chartqa/val,scienceqa/val')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='enable debugger')

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    if args.debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger('9.67.167.240', 12345)

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    gt_data = {}
    total_cnt = 0
    sub_ds_list = args.sub_ds_list.split(',')
    for ds in sub_ds_list:
        with open(os.path.join(args.data_path, ds, args.test_file_name), 'r') as f:
            ds_data = json.load(f)
        for c in tqdm(ds_data):
            if 'image' not in c:
                continue
            convs = c['conversations']
            for it in range(0, len(convs), 2):
                cur = {}

                assert convs[it]['from'] == 'human'
                assert convs[it + 1]['from'] == 'gpt'
                cur['question_type'] = 'short-answer'
                cur['ground_truth'] = convs[it + 1]['value']

                question = convs[it]['value'].replace('<image>\n', '').replace('<image>', '')  # .replace('<image>\n', '<image 1> ').replace('<image>', '<image 1>')
                answer = convs[it + 1]['value']

                base_id = ds + '_' + str(c['id'])
                for_hash = base_id + ' ' + question + ' ' + answer
                suffix = hashlib.md5(str(for_hash).encode('utf-8')).hexdigest()
                id = base_id + '_' + suffix

                if id in gt_data:
                    print(f'Repeated ID: {id}')
                gt_data[id] = cur
                total_cnt += 1

    print(f'Total processed: {total_cnt}, Total stored: {len(gt_data)}')

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_json(args.output_path, gt_data)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

