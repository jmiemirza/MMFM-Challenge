import json
import os
import random
import glob
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.main_data_dir = args.main_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'iconqa'
        self.split = ['train', 'val']

    def create_data(self):

        test_path_1 = os.path.join(f'{self.out_data_dir}_fill_in_blank', f'converted_output_test.json')
        with open(test_path_1, "r") as f:
            data = json.load(f)
        ids_list = [item["id"] for item in data]

        test_path_2 = os.path.join(f'{self.out_data_dir}_choose_txt', f'converted_output_test.json')
        with open(test_path_2, "r") as f:
            data = json.load(f)
        ids_list += [item["id"] for item in data]

        for i, split in enumerate(self.split):


            for answer_style in ['fill_in_blank', 'choose_txt']:


                target_format = []
                dataset_name = f'{self.dataset_name}_{answer_style}'
                instructions = load_instructions(self.instruction_path)[dataset_name]

                data_dir = os.path.join(self.data_dir, f'{split}/{answer_style}/*')

                for j, file_path in enumerate(glob.glob(data_dir)):

                    data_path = os.path.join(file_path, 'data.json')
                    image_path = os.path.join(file_path, 'image.png')
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    question = data['question']
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', question)
                    if answer_style == 'fill_in_blank':
                        value = data['answer']
                    else:
                        options = data['choices']

                        answer_index = data['answer']
                        value = str(options[answer_index])
                        instruction = instruction.replace('<options>', f': {options}')


                    instruction = '<image>\n' + instruction

                    if f'iconqa_{i}_{j}' in ids_list and split == 'val':
                        continue

                    file_name = image_path
                    image_relative_path = os.path.relpath(image_path, self.main_data_dir)
                    target_format.append({
                        'id': f'iconqa_{i}_{j}',
                        "image": image_relative_path,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': f"{value}"},
                        ],
                    })

                out_data_dir = f'{self.out_data_dir}_{answer_style}'
                out_filepath = os.path.join(out_data_dir, f'converted_output_{split}.json')
                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

                print(f'{split}: {len(target_format)}')
                with open(out_filepath, "w") as f:
                    json.dump(target_format, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/iconqa/iconqa_data/iconqa/', type=str)
    parser.add_argument('--main_data_dir', default='.', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/iconqa', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()