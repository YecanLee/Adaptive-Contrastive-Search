import json
import torch
import mauve 
import argparse
import numpy as np

def decode(tokens, tokenizer):
    token_id_list = tokenizer.convert_tokens_to_ids(tokens)
    text = tokenizer.decode(token_id_list)
    return text

def parse_text(reference_text, prediction_text, tokenizer):
    reference_tokens = tokenizer.tokenize(reference_text)
    prediction_tokens = tokenizer.tokenize(prediction_text)
    min_len = min(len(reference_tokens), len(prediction_tokens))
    reference_tokens = reference_tokens[:128]#reference_tokens[:min_len]
    reference_text = decode(reference_tokens, tokenizer)
    prediction_tokens = prediction_tokens[:128]#prediction_tokens[:min_len]
    prediction_text = decode(prediction_tokens, tokenizer)
    '''
        Truncating the text to maximum token of 128 accoridng to the author's implementation:
        https://github.com/XiangLi1999/ContrastiveDecoding/blob/98cad19349fb08ee95b0f25a661179866f8e2c84/text-generation/eval_script.py#L228

        Only evaluate instances with exact 128 tokens length based on the author's implementation:
        https://github.com/XiangLi1999/ContrastiveDecoding/blob/98cad19349fb08ee95b0f25a661179866f8e2c84/text-generation/eval_script.py#L235
    '''
    if min(len(reference_tokens), len(prediction_tokens)) == 128:
        flag = True
    else:
        flag = False
    return reference_text, prediction_text, flag

def load_result(in_f, tokenizer):
    with open(in_f) as f:
        result_list = json.load(f)

    # load reference list
    reference_list = []
    for item in result_list:
        one_reference_text = item['reference_text']
        reference_list.append(one_reference_text)

    # load all predictions
    number_of_predictions_per_instance = len(result_list[0]['generated_result'])
    print ('Number of predictions per instance is {}'.format(number_of_predictions_per_instance))
    all_prediction_list = []
    for idx in range(number_of_predictions_per_instance):
        one_prediction_list = []
        for item in result_list:
            one_prediction = item['generated_result'][str(idx)]
            one_prediction_list.append(one_prediction)
        assert len(one_prediction_list) == len(reference_list)
        all_prediction_list.append(one_prediction_list)
    return reference_list, all_prediction_list

def evaluate_one_instance(reference_list, prediction_list, tokenizer):
    ref_list, pred_list = [], []
    data_num = len(reference_list)
    for idx in range(data_num):
        one_ref, one_pred = reference_list[idx], prediction_list[idx]
        one_ref, one_pred, flag = parse_text(one_ref, one_pred, tokenizer)
        if flag:
            pass
        else:
            continue
        if len(one_pred.strip()) > 0: # igore predictions with zero length
            ref_list.append(one_ref)
            pred_list.append(one_pred)
            
    # use gpt2 model as the based model based on the author's implementation:
    # https://github.com/XiangLi1999/ContrastiveDecoding/blob/98cad19349fb08ee95b0f25a661179866f8e2c84/text-generation/eval_script.py#L248
    out =  mauve.compute_mauve(p_text=ref_list, q_text=pred_list, device_id=0, verbose=False,
        featurize_model_name='gpt2')
    mauve_score = out.mauve
    return mauve_score*100

def measure_mauve(in_f):
    from transformers import AutoTokenizer
    

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    reference_list, all_prediction_list = load_result(in_f, tokenizer)

    mauve_score_list = []
    for idx in range(len(all_prediction_list)):
        one_prediction_list = all_prediction_list[idx]
        one_mauve_score = evaluate_one_instance(reference_list, one_prediction_list, tokenizer)
        mauve_score_list.append(one_mauve_score)

    mean, std = round(np.mean(mauve_score_list),2), round(np.std(mauve_score_list),2)
    result_dict = {
        "mauve_score_list": [str(num) for num in mauve_score_list],
        'mauve_mean': str(mean),
        'mauve_std': str(std)
    }
    return result_dict
