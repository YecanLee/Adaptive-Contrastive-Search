import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import json
import torch
import numpy as np
import logging
import argparse
from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel, AutoTokenizer
from helpers.process_data import load_data


logging.getLogger('transformers.generation_utils').disabled = True

# A fast inference setting for Ampere GPUs
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('Fast inference setting for Ampere GPUs is enabled 🔥🔥🔥.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Config file path')
    parser.add_argument('--data_num', type=int, default=None, help='Number of Prefix used for text generation')
    parser.add_argument('--k', type=int, default=None, help='CS hyperparameter k', required=True)
    parser.add_argument('--alpha', type=float, default=None, help='CS hyperparameter alpha', required=True)
    parser.add_argument('--save_path_prefix', type=str, default=None, help='Path to save the generation results')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset')
    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else 'cpu')

    config = OmegaConf.load(args.config_path)

    args.save_path_prefix = config.experiments.save_path_prefix if args.save_path_prefix is None else args.save_path_prefix
    args.model_name = config.experiments.model_name if args.model_name is None else args.model_name
    args.dataset = config.experiments.dataset if args.dataset is None else args.dataset
    args.k = config.experiments.k if args.k is None else args.k
    print(f'Using model {args.model_name} for story generation now!')
    print(f'Using dataset {args.dataset} as prefix prompt now!')

    assert args.dataset in ['book', 'wikinews', 'wikitext'], "Dataset must be one of 'book', 'wikinews', or 'wikitext'"
    full_data_path = f'{config.experiments.dataset_prefix}/{args.dataset}_contrastive_{args.model_name}_256.jsonl'
    print(f'Full data path is {full_data_path}')

    save_path_prefix = f'{args.save_path_prefix}/{args.model_name}/{args.dataset}/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = f'{args.dataset}_static_contrastive_k_{args.k}_alpha_{args.alpha}_result.json'
    save_path = os.path.join(save_path_prefix, save_name)
    print(f'Result saving path is {save_path}')

    print('Loading model...🔨🔨🔨')
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    pad_token = '<_PAD_>'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = pad_token
    special_tokens_dict = {'pad_token': pad_token}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model = model.to(device)
    model = torch.compile(model)
    model.eval()
    print('Model loaded.')

    prefix_text_list, prefix_token_id_list, reference_text_list = load_data(full_data_path, tokenizer, mode=args.dataset)

    print('Performing inference...🔥🔥🔥')
    data_num = len(prefix_text_list) if args.data_num is None else args.data_num
    print(data_num)
    result_list = []

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=cuda_available, dtype=torch.float16, cache_enabled=True):
            for index in range(data_num):
                if args.data_num:
                    print(f'Inference {index + 1}/{args.data_num} ({np.round((index + 1)/args.data_num*100, 2)} %)')
                else:
                    print(f'Inference {index + 1}/{data_num} ({np.round((index + 1)/data_num*100, 2)} %)')
                one_prefix_text = prefix_text_list[index]
                one_reference_text = reference_text_list[index]
                input_ids = tokenizer(one_prefix_text, return_tensors='pt').input_ids
                if cuda_available:
                    input_ids = input_ids.cuda(device)
                _, prefix_len = input_ids.size()
                
                output = model.generate(input_ids, 
                                        penalty_alpha=args.alpha, 
                                        top_k=args.k, 
                                        max_length=prefix_len+256)

                one_generation_text = tokenizer.decode(output[0][prefix_len:], skip_special_tokens=True)
                              
                one_res_dict = {
                    'prefix_text': one_prefix_text,
                    'reference_text': one_reference_text,
                    'generated_result': {
                        '0': one_generation_text
                    }
                }
                result_list.append(one_res_dict)

        print('Inference completed! 🥳🥳🥳')
        with open(save_path, 'w') as outfile:
            json.dump(result_list, outfile, indent=4)
