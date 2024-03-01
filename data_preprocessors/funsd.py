import json
import os
import random
import cv2
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
        self.dataset_name = 'funsd'
        self.split = ['training', 'testing']
        self.label_mapping = {'header': 'title',
                              'question': 'key',
                              'answer': 'value',
                              'other': 'other'}

    def create_data(self):
        instructions = load_instructions(self.instruction_path)[self.dataset_name]
        test_path = os.path.join(self.out_data_dir, f'converted_output_test.json')
        with open(test_path, "r") as f:
            data = json.load(f)
        ids_list = [item["id"] for item in data]
        for i, split in enumerate(self.split):
            target_format = []
            ann_dir = os.path.join(self.data_dir, f'{split}_data/annotations')
            img_dir = os.path.join(self.data_dir, f'{split}_data/images')
            for j, file in enumerate(tqdm(sorted(os.listdir(ann_dir)))):
                file_path = os.path.join(ann_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_path = os.path.join(img_dir, file)
                image_path = image_path.replace('.json', '.png') 
                image = cv2.imread(image_path)
                h, w, _ = image.shape

                items = []
                for k, item in enumerate(data["form"]):
                    text = item['text']
                    words, label = item["words"], item["label"]
                    label = self.label_mapping[label]
                    words = [w for w in words if w["text"].strip() != ""]
                    if len(words) == 0:
                        continue
                    start_bbox, end_bbox = words[0]['box'], words[-1]['box']
                    bbox = [start_bbox[0], start_bbox[1], end_bbox[2], start_bbox[3]]
                    bbox = normalize_bbox(bbox, w, h)
                    items.append((text, label, bbox))
                items = sort_coordinate(items)

                text_sequence = []
                bboxes = []
                labels = {}
                for l, item in enumerate(items):
                    text, label, bbox = item
                    labels[text] = label
                    text_sequence.append(text)
                    bbox = [bbox] * len(text)
                    bboxes += bbox

                ocr = ' '.join(text_sequence)
                for m, key in enumerate(labels):
                    instruction = random.choice(instructions)
                    instruction = instruction.replace('<key>', key)
                    instruction = '<image>\n' + instruction

                    value = labels[key]

                    file_name = image_path
                    image_relative_path = os.path.relpath(image_path, self.main_data_dir)

                    if f'funsd_{i}_{j}_{k}_{l}_{m}' in ids_list and split == 'testing':
                        continue

                    target_format.append({
                        "id": f"funsd_{i}_{j}_{k}_{l}_{m}",
                        "image": image_relative_path,
                        # "ocr": ocr,
                        # "bboxes": bboxes,
                        "conversations": [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': value},
                        ],
                    })

            split = split.replace('ing', '')

            if split == 'test':
                split = 'val'


            out_filepath = os.path.join(self.out_data_dir, f'converted_output_{split}.json')
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            with open(out_filepath, "w") as f:
                json.dump(target_format, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='raw_datasets/funsd/dataset', type=str)
    parser.add_argument('--main_data_dir', default='.', type=str)
    parser.add_argument('--out_data_dir', default='processed_data/funsd', type=str)
    args = parser.parse_args()
    
    dataset = InstructData(args)
    dataset.create_data()