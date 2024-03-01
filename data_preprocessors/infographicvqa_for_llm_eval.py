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
        self.dataset_name = 'infographicvqa'
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
        for i, split in enumerate(self.split):
            if split == 'dev':
                file_name = os.path.join(self.data_dir, split, 'document.jsonl')
                with open(file_name, 'r') as f:
                    data = f.readlines()

                # ocrs = self.create_ocr_data(split)
                target_format = []
                target_dict_of_dict = {}
                for j, d in enumerate(tqdm(data)):
                    d = json.loads(d)

                    image_name = d['name']

                    file_name = os.path.join(self.img_dir, image_name + '.jpg')  # hard coded, until we find a better way
                    image_relative_path = os.path.relpath(file_name, self.main_data_dir)

                    if not os.path.exists(file_name):
                        print(f'File not found found: {file_name} -- SKIPPING')
                        continue

                    statfile = os.stat(file_name)
                    if statfile.st_size == 0:
                        print(f'File is empty/corrupted: {file_name} -- SKIPPING')
                        continue

                    # file_name = os.path.join(self.data_dir, 'png', image_name, '0.jpg')
                    # file_name = file_name

                    for k, ann in enumerate(d['annotations']):
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

                        target_dict_of_dict[f'infographicvqa_{i}_{j}_{k}'] = {
                            'id': f'infographicvqa_{i}_{j}_{k}',
                            "image": image_relative_path,
                            "conversations": [
                                {'from': 'human', 'value': instruction},
                                {'from': 'gpt', 'value': value},
                            ],
                        }

        test_file_list = []
        for item_id in ids_list:
            test_file_list.append(target_dict_of_dict[item_id])

        out_filepath = os.path.join(self.out_data_dir.replace('processed_data', 'processed_data_for_mixtral_eval'), f'converted_output_test.json')
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)



        print(f'test: {len(test_file_list)}')
        with open(out_filepath, "w") as f:
            json.dump(test_file_list, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/infovqa/aws_neurips_time/infographics_vqa', type=str)
    parser.add_argument('--main_data_dir', default='.', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/infographicvqa', type=str)
    parser.add_argument('--img_data_dir', default='raw_datasets/infovqa/jpgs/', type=str)
    parser.add_argument('--remove_instruct_templates', action='store_true')
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()