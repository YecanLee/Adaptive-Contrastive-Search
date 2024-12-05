import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers.generation.logits_process import ACSLogitsWarper
import torch
import numpy as np
import os
import json
from tqdm import trange
from helpers.process_data import load_data

# A fast inference setting for Ampere GPUs
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('Fast inference setting for Ampere GPUs is enabled 🔥🔥🔥.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument('--dataset_prefix', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--save_path_prefix', type=str, default='mistralv03')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--q', required=True, type=float)
    parser.add_argument('--penalty', type=float, default=1.2)
    parser.add_argument('--save_file', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.cuda}' if cuda_available else 'cpu')

    assert args.dataset in ['book', 'wikinews', 'wikitext'], "Dataset must be one of 'book', 'wikinews', or 'wikitext'"
    full_data_path = f'{args.dataset_prefix}/{args.dataset}_contrastive_gpt2-xl_256.jsonl'
    print(f'Full data path is {full_data_path}')

    save_path_prefix = f'{args.save_path_prefix}/{args.dataset}/'
    print(f"Save path prefix is {save_path_prefix}")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = f'{args.dataset}_{args.save_file}_q_{args.q}_penalty_{args.penalty}.json'
    save_path = os.path.join(save_path_prefix, save_name)
    print(f'Result saving path is {save_path}')

    print('Loading model... 🔨🔨🔨')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cpu")
    model = torch.compile(model)
    model.to(device)

    # Initialize the dynamic warper used in the iACS method
    dynamic_warper = ACSLogitsWarper(
        pad_token_id=tokenizer.pad_token_id,
        q=args.q,
        penalty=args.penalty
    )

    logits_processor = LogitsProcessorList([dynamic_warper])
    prefix_text_list, prefix_token_id_list, reference_text_list = load_data(full_data_path, tokenizer, mode=args.dataset)

    print('Performing inference 🚀🚀🚀...')
    data_num = len(prefix_text_list)
    print(data_num)
    result_list = []
    batch_size = args.batch_size
    max_len = max(len(tokenizer.encode(text)) for text in prefix_text_list)

    with torch.inference_mode():
        for index in trange(0, data_num, batch_size, desc='Inferring... ⌛⌛⌛'):
            torch.cuda.synchronize()
            batch_prefix_text = prefix_text_list[index:index+batch_size]
            batch_reference_text = reference_text_list[index:index+batch_size]
            model_inputs = tokenizer(batch_prefix_text, padding='max_length', padding_side="left", max_length=max_len, truncation=False, return_tensors="pt").to(device)
            generated_ids = model.generate(**model_inputs, logits_processor=logits_processor, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
            batch_generation_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)        
            
            current_batch_size = min(batch_size, data_num - index)
            for i in range(current_batch_size):
                one_res_dict = {
                    'prefix_text': batch_prefix_text[i],
                    'reference_text': batch_reference_text[i],
                    'generated_result': {
                        '0': batch_generation_text[i][len(batch_prefix_text[i]):]
                    }
                }
                result_list.append(one_res_dict)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print('Inference completed! 🎉🎉🎉')

        with open(save_path, 'w') as outfile:
            json.dump(result_list, outfile, indent=4)