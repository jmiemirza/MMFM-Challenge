import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

# sub_ds_list = ['llavar', 'docvqa', 'infographicsvqa', 'chartqa/train', 'scienceqa/train']
# train_file_name = 'converted_output_train.json'

# added_ds = ['docvqa', 'funsd', 'iconqa_choose_txt', 'iconqa_fill_in_blank', 'infographicvqa', 'tabfact', 'textbookqa', 'websrc', 'wildreceipt', 'wtq'] # ['ai2d', 'deepform', 'doclaynet', 'docvqa', 'funsd', 'iconqa_choose_txt', 'iconqa_fill_in_blank', 'infographicvqa', 'klc', 'pwc', 'tabfact', 'textbookqa', 'wildreceipt', 'wtq']
#
# sub_ds_list = ['processed_data/' + ds for ds in added_ds] # ['chartqa/train', 'scienceqa/train'] +
# train_file_name = 'converted_output_train.json'

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./data/llava_doc_vl_mix.json', help='name of saved json')
    parser.add_argument('--data_path', type=str, default="/dccstor/leonidka1/irenespace/mmfm_challenge/data")

    parser.add_argument('--train_file_name', type=str, default='converted_output_train.json')
    parser.add_argument('--sub_ds_list', type=str, default='docvqa,infographicvqa,websrc,wtq,iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact')

    parser.add_argument('--mixin', type=str, default=None, help='if not None will mix in additional data')
    parser.add_argument('--debug', action='store_true', help='enable debugger')

    args = parser.parse_args()

    if args.debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger('9.67.169.241', 12345)

    base_path = os.path.dirname(args.output_path)
    os.makedirs(base_path, exist_ok=True)

    all_data = []
    sub_ds_list = args.sub_ds_list.split(',')
    train_file_name = args.train_file_name
    for ds in sub_ds_list:
        tr_json_pth = os.path.join(args.data_path, ds, train_file_name)
        with open(tr_json_pth, 'r') as f:
            d = json.load(f)
        assert isinstance(d, list)
        for c in tqdm(d, desc=ds):
            if 'image' in c:
                c['image'] = os.path.join(ds, c['image']) # args.data_path will be images root path for LLaVA train
        all_data.extend(d)

    if args.mixin is not None:
        with open(args.mixin, 'r') as f:
            d = json.load(f)
        all_data.extend(d)

    with open(args.output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

if __name__ == '__main__':
    main()