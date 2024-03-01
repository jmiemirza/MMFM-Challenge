import json
import os
import random
import argparse
import csv

from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, load_instructions
from collections import defaultdict
from google_vision_ocr import Google_OCR

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.main_data_dir = args.main_data_dir
        self.out_data_dir = args.out_data_dir
        # self.ocr_dir = os.path.join(args.input_data_dir, 'ocrs')
        self.dataset_name = 'websrc'
        self.google_ocr = Google_OCR(args.api_key)
        self.split = ['train', 'dev']
        # os.makedirs(self.ocr_dir, exist_ok=True)
        self.remove_instruct_templates = args.remove_instruct_templates
    
    def load_split_info(self):
        file_name = os.path.join(self.data_dir, 'dataset_split.csv')
        with open(file_name) as f:
            reader = csv.reader(f)
            split_info = defaultdict(list)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                number = '0' + row[1] if int(row[1]) < 10 else  row[1]
                split = row[3]
                data_path = os.path.join(self.data_dir, f'{row[0]}/{number}/dataset.csv')
                split_info[split].append(data_path)
        return split_info
        
    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]

        test_path = os.path.join(self.out_data_dir, f'converted_output_test.json')
        with open(test_path, "r") as f:
            data = json.load(f)

        ids_list = [item["id"] for item in data]
        split_info = self.load_split_info()
        for ii, split in enumerate(self.split):
            target_format = []
            for j, data_path in enumerate(tqdm(split_info[split])):
                with open(data_path) as f:
                    data_dir = os.path.dirname(data_path)
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i == 0:
                            for index, element in enumerate(row):
                                if 'question' == element:
                                    question_index = index
                                elif 'id' == element:
                                    id_index = index
                                elif 'answer' == element:
                                    answer_index = index
                            continue   
                        questionId = row[id_index]

                        image_path = os.path.join(data_dir, f'processed_data/{questionId[2:9]}.png')
                        image_relative_path = os.path.relpath(image_path, self.main_data_dir)


                        if not os.path.exists(image_path):
                            print(f'File not found found: {image_path} -- SKIPPING')
                            continue

                        statfile = os.stat(image_path) # very LAZY check for empty file
                        if statfile.st_size == 0:
                            print(f'File is empty/corrupted: {image_path} -- SKIPPING')
                            continue

                        question = row[question_index]
                        if self.remove_instruct_templates:
                            instruction = question
                        else:
                            instruction = random.choice(instructions)
                            instruction = instruction.replace('<key>', question)
                            instruction = '<image>\n' + instruction


                        value = row[answer_index]


                        if f'websrc_{ii}_{j}_{i}' in ids_list and split == 'dev':
                            continue

                        target_format.append({
                            'id': f'websrc_{ii}_{j}_{i}',
                            "image": image_relative_path,
                            "conversations": [
                                {'from': 'human', 'value': instruction},
                                {'from': 'gpt', 'value': value},
                            ],
                        })

            if split == 'dev': # LAZY fix
                split = 'val'

            out_filepath = os.path.join(self.out_data_dir, f'converted_output_{split}.json')
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')

            with open(out_filepath, "w") as f:
                json.dump(target_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/websrc/', type=str)
    parser.add_argument('--main_data_dir', default='.', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/websrc', type=str)
    parser.add_argument('--ocr_dir', default='raw_datasets/websrc/ocrs', type=str)
    parser.add_argument('--api_key', default='API_KEY', type=str)
    parser.add_argument('--remove_instruct_templates', action='store_true')
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()