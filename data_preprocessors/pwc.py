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
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'pwc'
        self.split = ['train', 'dev']

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
            break
        return ocrs

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        for i, split in enumerate(self.split):
            file_name = os.path.join(self.data_dir, split, 'document.jsonl')
            with open(file_name, 'r') as f:
                data = f.readlines()
            target_format = []
            for j, d in enumerate(tqdm(data)):
                d = json.loads(d)
                image_name = d['name']

                file_name = os.path.join('./data/raw_datasets/pwc/', 'jpgs',
                                         image_name + '.jpg')  # hard coded, until we find a better way

                if not os.path.exists(file_name):
                    print(f'File not found found: {file_name} -- SKIPPING')
                    continue

                statfile = os.stat(file_name)
                if statfile.st_size == 0:
                    print(f'File is empty/corrupted: {file_name} -- SKIPPING')
                    continue
                file_name = file_name
                for k, ann in enumerate(d['annotations']):
                    instruction = random.choice(instructions)
                    if 'children' in ann['values'][0]:
                        for l, v in enumerate(ann['values']):
                            for m, child in enumerate(v['children']):
                                value = child['key']
                                key = child['values'][0]['value']
                                instruction = instruction.replace('<key>', key)

                                instruction = '<image>\n' + instruction

                                # ocr, bboxes = ocrs[image_name][0], ocrs[image_name][1]

                                target_format.append({
                                    'id': f'pwc_{i}_{j}_{k}_{l}_{m}',
                                    "image": file_name,
                                    "conversations": [
                                        {'from': 'human', 'value': instruction},
                                        {'from': 'gpt', 'value': value},
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
    parser.add_argument('--input_data_dir', default='raw_datasets/pwc/aws_neurips_time/AxCell', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/pwc', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()