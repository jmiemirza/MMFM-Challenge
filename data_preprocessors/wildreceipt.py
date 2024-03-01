import json
import os
import random
from PIL import Image, ImageSequence
from tqdm import tqdm 
from pathlib import Path
from utils import normalize_bbox, sort_coordinate, load_instructions
import argparse

class InstructData:
    def __init__(self, args):
        self.instruction_path = Path('instructdoc_instructions.xlsx')
        self.data_dir = args.input_data_dir
        self.main_data_dir = args.main_data_dir
        self.out_data_dir = args.out_data_dir
        self.dataset_name = 'wildreceipt'
        self.split = ['train', 'test']
        self.classes = {}
        for items in open(os.path.join(args.input_data_dir, 'class_list.txt')):
            index, label = items.split()
            self.classes[index] = label

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        test_path = os.path.join(self.out_data_dir, f'converted_output_test.json')
        with open(test_path, "r") as f:
            data = json.load(f)
        ids_list = [item["id"] for item in data]
        for i, split in enumerate(self.split):
            target_format = []
            with open(os.path.join(self.data_dir, f'{split}.txt')) as f:
                samples = f.readlines()
            for j, sample in enumerate(tqdm(samples)):
                data = json.loads(sample)
                file_name = data['file_name']
                image_path = os.path.join(self.data_dir, file_name)
                image = Image.open(image_path)
                w, h = image.size

                items = []
                labels = {}
                for k, item in enumerate(data["annotations"]):
                    text, label_index = item["text"], item["label"]
                    label = self.classes[str(label_index)]
                    if label_index == 0:
                        continue
                    bbox = item["box"]
                    bbox = [bbox[0], bbox[1], bbox[4], bbox[5]]
                    bbox = normalize_bbox(bbox, w, h)
                    items.append((text, label, bbox))

                items = sort_coordinate(items)

                ocr = []
                bboxes = []
                for l, item in enumerate(items):
                    words, label, bbox = item
                    labels[words] = label
                    ocr.append(words)
                    bbox = [bbox] * len(words.split())
                    bboxes += bbox
                ocr = ' '.join(ocr)

                for m, key in enumerate(labels):
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', key)
                    value = labels[key]
                    instruction = '<image>\n' + instruction



                    if f'wildreceipt_{i}_{j}_{k}_{l}_{m}' in ids_list and split == 'test':
                        continue

                    file_name = image_path
                    image_relative_path = os.path.relpath(image_path, self.main_data_dir)
                    target_format.append({
                        "id": f'wildreceipt_{i}_{j}_{k}_{l}_{m}',
                        "image":  image_relative_path,
                        # "ocr": ocr,
                        # "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': value},
                        ],
                    })
            if split == 'test':
                split = 'val'
            out_filepath = os.path.join(self.out_data_dir, f'converted_output_{split}.json')
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            with open(out_filepath, "w") as f:
                json.dump(target_format, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/wildreceipt/wildreceipt', type=str)
    parser.add_argument('--main_data_dir', default='.', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/wildreceipt', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()