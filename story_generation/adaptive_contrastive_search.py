import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import argparse
import json
import torch
import os
import numpy as np
import logging
from transformers import GPT2Tokenizer
from omegaconf import OmegaConf
import story_generation.simctg_acs_base as simctg_acs_base
from helpers.process_data import load_data
from tqdm import trange

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
    parser.add_argument('--q', type=int, default=None, help='ACS hyperparameter q')
    parser.add_argument('--k', type=int, default=None, help='ACS initialization k value')
    parser.add_argument('--save_path_prefix', type=str, default=None, help='Path to save the generation results')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum length of the generated text')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset')
    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda_available else 'cpu')
    
    config = OmegaConf.load(args.config_path)
    args.q = config.model.q if args.q is None else args.q
    args.k = config.experiments.k if args.k is None else args.k
    args.save_path_prefix = config.experiments.save_path_prefix if args.save_path_prefix is None else args.save_path_prefix
    args.model_name = config.experiments.model_name if args.model_name is None else args.model_name
    args.dataset = config.experiments.dataset if args.dataset is None else args.dataset
    print(f'Initializing ACS with k = {args.k} at the beginning!')
    print(f'Using model {args.model_name} for story generation now!')
    print(f'Using dataset {args.dataset} as prefix prompt now!')
    print(f'Using q = {args.q} as ACS hyperparameter now!')

    assert config.experiments.dataset in ['book', 'wikinews', 'wikitext'], "Dataset must be one of 'book', 'wikinews', or 'wikitext'"
    full_data_path = f'{config.experiments.dataset_prefix}/{args.dataset}_contrastive_{args.model_name}_256.jsonl'
    print(f'Full data path is {full_data_path}')

    save_path_prefix = f'{config.experiments.save_path_prefix}/{args.model_name}/{args.dataset}/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = f'{args.dataset}_adaptive_contrastive_search_k_{args.k}_alpha_{config.experiments.alpha}_result.json'
    save_path = os.path.join(save_path_prefix, save_name)
    print(f'Result saving path is {save_path}')

    print('Loading model...🔨🔨🔨')
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    eos_token_id = tokenizer.eos_token_id
    pad_token = '<_PAD_>'
    model = simctg_arctanh_base.SimCTG(args.model_name, pad_token)
    if cuda_available:
        model = model.to(device)
    model = torch.compile(model,mode='max-autotune')
    model.eval()
    print('Model loaded.')

    prefix_text_list, prefix_token_id_list, reference_text_list = load_data(full_data_path, tokenizer, mode=args.dataset)

    print('Performing inference...🤗🤗🤗')
    data_num = len(prefix_text_list) if args.data_num is None else args.data_num
    print(f"Generating results for {data_num} samples...")
    result_list = []

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=cuda_available, dtype=torch.float16, cache_enabled=True):
            for index in trange(data_num):
                one_prefix_text = prefix_text_list[index]
                one_reference_text = reference_text_list[index]
                input_ids = tokenizer(one_prefix_text, return_tensors='pt').input_ids
                if cuda_available:
                    input_ids = input_ids.cuda(device)
                _, prefix_len = input_ids.size()
                
                output = model.fast_contrastive_search(input_ids,
                                                        beam_width = args.k,
                                                        alpha=config.experiments.alpha,
                                                        decoding_len=args.max_len,
                                                        args=args)
                
                one_generation_text = tokenizer.decode(output[prefix_len:], skip_special_tokens=True)                          
                
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
