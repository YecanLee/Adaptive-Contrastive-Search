import json

def parse_text(item, tokenizer, mode):
    if mode == 'wikitext':
        prefix_text = item[0]['prompt'].strip(' ')
        full_text = item[0]['gold_ref'].strip(' ')
    else:
        prefix_text = item[0]['prompt']
        full_text = item[0]['gold_ref']
    prefix_token_list = tokenizer.tokenize(prefix_text)
    prefix_token_id_list = tokenizer.convert_tokens_to_ids(prefix_token_list)
    prefix_len = len(prefix_token_id_list)

    full_token_list = tokenizer.tokenize(full_text)
    full_token_id_list = tokenizer.convert_tokens_to_ids(full_token_list)
    reference_text = tokenizer.decode(full_token_id_list[prefix_len:])
    return prefix_text, prefix_token_id_list, reference_text

def load_data(in_f, tokenizer, mode):
    with open(in_f, 'r') as json_file:
        json_list = list(json_file)

    result_list = [json.loads(json_str) for json_str in json_list]
    
    prefix_text_list, prefix_token_id_list, reference_text_list = [], [], []
    for item in result_list:
        one_prefix_text, one_prefix_token_id, one_reference_text = parse_text(item, tokenizer, mode)
        prefix_text_list.append(one_prefix_text)
        prefix_token_id_list.append(one_prefix_token_id)
        reference_text_list.append(one_reference_text)
    return prefix_text_list, prefix_token_id_list, reference_text_list
