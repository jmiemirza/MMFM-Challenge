import os
import json
import random
import argparse

train_val_datasets = [
    'docvqa', 'funsd', 'iconqa_choose_txt', 'iconqa_fill_in_blank',
                       'infographicvqa',  'tabfact', 'textbookqa', 'wildreceipt', 'wtq',
                       'websrc'
                       ]


def merge_datasets(input_data_dir='./processed_data', save_dir='./', max_samples=5000):
    questionId = 0
    for split in ['train', 'val', 'test']:
        merge = []
        for dataset_name in train_val_datasets:
            dataset_path = os.path.join(input_data_dir, dataset_name, f'converted_output_{split}.json')

            if os.path.exists(dataset_path):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
            else:
                print(f'{dataset_path} do not exist')
                continue
            print(f'{dataset_name} : {split}: {len(data)}')
            for d in data:
                merge.append(d)
        random.shuffle(merge)

        out_filepath = os.path.join(save_dir, f'converted_output_{split}.json')
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        print(f'Total {split} Samples: {len(merge)}')
        with open(out_filepath, "w") as f:
            json.dump(merge, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', default='processed_data', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--max_samples', default=5000, type=int)
    args = parser.parse_args()

    merge_datasets(args.input_data_dir, args.save_dir, args.max_samples)