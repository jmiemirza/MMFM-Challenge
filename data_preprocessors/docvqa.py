import json
import os
import random
from PIL import Image, ImageSequence
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
        self.img_dir = args.img_data_dir
        self.dataset_name = 'docvqa'
        self.split = ['train', 'dev']
        self.remove_instruct_templates = args.remove_instruct_templates

    def create_ocr_data(self, split):
        file_name = os.path.join(self.data_dir, split, 'documents_content.jsonl')
        with open(file_name, 'r') as f:
            data = f.readlines()
        ocrs = {}
        for d in data:
            d = json.loads(d)
            image_name = d['name'].replace('.pdf', '')
            try:
                content = d['contents'][1] # microsoft cv
            except:
                content = d['contents'][0] # tesseract

            bboxes = []
            tokens = []
            try:
                _ , _, w, h = content['common_format']['structures']['pages']['positions'][0]
                for token, bbox in zip(content['common_format']['tokens'], content['common_format']['positions']):
                    bbox = normalize_bbox(bbox, w, h)
                    bboxes.append(bbox)
                    tokens.append(token)
            except:
                pass
            ocrs[image_name] = (' '.join(tokens), bboxes)
        return ocrs

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        test_path = os.path.join(self.out_data_dir, f'converted_output_test.json')
        with open(test_path, "r") as f:
            data = json.load(f)
        ids_list = [item["id"] for item in data]
        for split in self.split:
            file_name = os.path.join(self.data_dir, split, 'document.jsonl')
            with open(file_name, 'r') as f:
                data = f.readlines()

            # ocrs = self.create_ocr_data(split)
            target_format = []
            for i, d in enumerate(tqdm(data)):
                d = json.loads(d)

                image_name = d['name']

                file_name = os.path.join(self.img_dir, image_name + '.jpg') # hard coded, until we find a better way
                image_relative_path = os.path.relpath(file_name, self.main_data_dir)

                if not os.path.exists(file_name):
                    print(f'File not found found: {file_name} -- SKIPPING')
                    continue

                statfile = os.stat(file_name)
                if statfile.st_size == 0:
                    print(f'File is empty/corrupted: {file_name} -- SKIPPING')
                    continue

                for j, ann in enumerate(d['annotations']):
                    question = ann['key']
                    if self.remove_instruct_templates:
                        instruction = question
                    else:
                        instruction = random.choice(instructions)
                        instruction = instruction.replace('<key>', question)
                        instruction = '<image>\n' + instruction

                    bboxes = []
                    # ocr, bboxes = ocrs[image_name][0], ocrs[image_name][1]
                    value = ann['values'][0]['value']
                    values = ann['values'][0]['value_variants']

                    if f'docvqa_{i}_{j}' in ids_list and split == 'dev':
                        continue

                    target_format.append({
                        'id': f"docvqa_{i}_{j}",
                        "image": image_relative_path,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': value,},
                        ],
                    })

            if split == 'dev':
                split = 'val'

            out_filepath = os.path.join(self.out_data_dir, f'converted_output_{split}.json')
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            print(f'{split}: {len(target_format)}')
            with open(out_filepath, "w") as f:
                json.dump(target_format, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/docvqa/aws_neurips_time/docvqa', type=str)
    parser.add_argument('--main_data_dir', default='.', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/docvqa', type=str)
    parser.add_argument('--img_data_dir', default='raw_datasets/docvqa/jpgs/', type=str)
    parser.add_argument('--remove_instruct_templates', action='store_true')
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()