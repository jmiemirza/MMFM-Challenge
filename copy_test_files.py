import os
from argparse import ArgumentParser
import shutil
from tqdm import tqdm

added_ds = ['docvqa', 'funsd', 'iconqa_choose_txt', 'iconqa_fill_in_blank', 'infographicvqa', 'tabfact', 'textbookqa', 'websrc', 'wildreceipt', 'wtq'] # ['ai2d', 'deepform', 'doclaynet', 'docvqa', 'funsd', 'iconqa_choose_txt', 'iconqa_fill_in_blank', 'infographicvqa', 'klc', 'pwc', 'tabfact', 'textbookqa', 'wildreceipt', 'wtq']

sub_ds_list = ['processed_data/' + ds for ds in added_ds] # ['chartqa/train', 'scienceqa/train'] +
test_file_name = 'converted_output_test.json'

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./data/test', help='name of saved json')
    parser.add_argument('--data_path', type=str, default="/dccstor/leonidka1/irenespace/mmfm_challenge/data")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    for ds in tqdm(sub_ds_list):
        os.makedirs(os.path.join(args.output_path, ds), exist_ok=True)
        json_src_pth = os.path.join(args.data_path, ds, test_file_name)
        json_trg_pth = os.path.join(args.output_path, ds, test_file_name)
        shutil.copyfile(json_src_pth, json_trg_pth)


if __name__ == '__main__':
    main()