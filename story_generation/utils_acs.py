import os
# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up the relative import 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from helpers.process_data import load_data  
import torch
import torch.nn.functional as F
import random
import numpy as np
from scipy.stats import entropy
from omegaconf import OmegaConf
config = OmegaConf.load("configs/adaptive_contrastive_base.yaml")

alpha_list = []
kld_uniform_list = []
kld_cert_list = []
var_list = []
var_k_list = []
probability_list = []
perplex_list = []
token_list = []
k_list = []


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def alpha_update(var_out, var_max, var_list, q):
    # delta = kld_uniform - kld_max
    if len(var_list) == 0:
        # threshold = 0.5 * kld_max
        threshold = var_out
    else:
        threshold = np.median(var_list)
    var_centered = (var_out - threshold)/var_max 
    # q = 8
    sigmoid_arg = (q/2) * (np.log((1 + var_centered) / (1 - var_centered)))
    return  np.round(np.exp(sigmoid_arg)/(1 + np.exp(sigmoid_arg)), 2) 


def k_update(var_out, var_max, var_list, q):
    # delta = kld_uniform - kld_max
    if len(var_list) == 0:
        # threshold = 0.5 * kld_max
        threshold = var_out
    else:
        threshold = np.median(var_list)
    var_centered = (var_out - threshold) / (var_max)
    # q = 8
    sigmoid_arg = (q/2) * (np.log((1 + var_centered) / (1 - var_centered)))
    # delta = sign*np.exp(np.abs(var_centered))
    #delta = (1 + var_centered) ** (q / 2) / ((1 - var_centered) ** (q / 2) + (1 - var_centered) ** (q / 2))
    return int(np.round((np.exp(sigmoid_arg)/(1 + np.exp(sigmoid_arg))) * 10 + 5, 0)) 


def adaptive_k(logits, config):
    uniform_dist = [1/logits.shape[1] for i in range(0,logits.shape[1])]
    probs = [F.softmax(element, dim = 0).tolist() for element in logits][0]
    var_max = entropy(uniform_dist)
    var_out = entropy(probs)
    beam_width = k_update(var_out, var_max, var_list, q=config.q)

    return beam_width, var_out


def adaptive_alpha(logits, beam_width, config):
    uniform_dist = [1/beam_width for i in range(0,beam_width)]
    # Compute distances
    probs = [F.softmax(element, dim = 0).tolist() for element in logits][0]
    var_max_k = entropy(uniform_dist)
    probs_k = sorted(probs, reverse=True)[:beam_width]
    var_out_k = entropy(probs_k/np.sum(probs_k))
    alpha = alpha_update(var_out_k, var_max_k, var_k_list, q=config.q)

    return alpha, var_out_k
    

def ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, alpha):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    assert cosine_matrix.size() == torch.Size([beam_width, context_len])
    scores, _ = torch.max(cosine_matrix, dim = -1)
    assert scores.size() == torch.Size([beam_width])
    next_top_k_probs = next_top_k_probs.view(-1)
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    _, selected_idx = torch.topk(scores, k = 1)
    assert selected_idx.size() == torch.Size([1])
    selected_idx = selected_idx.unsqueeze(0)
    assert selected_idx.size() == torch.Size([1,1])
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    assert next_id.size() == torch.Size([1,1])
    
    return next_id


def ContrastiveDecodingOneStep(model, input_ids, beam_width, alpha):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    prev_hidden_states, logits = model.compute_logits_and_hidden_states(input_ids)
    _, seqlen, embed_dim = prev_hidden_states.size()
    _, _, vocab_size = logits.size()
    p = random.uniform(0, 1)

    logit_for_next_step = logits[:,-1,:]
    assert logit_for_next_step.size() == torch.Size([1, vocab_size])

    next_probs = F.softmax(logit_for_next_step, dim = -1)
    assert next_probs.size() == logit_for_next_step.size()

    _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
    assert top_k_ids.size() == torch.Size([1, beam_width])
        
    top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

    assert top_k_probs.size() == top_k_ids.size()
    # compute new hidden 
    expanded_context = [input_ids for _ in range(beam_width)]
    expanded_context = torch.cat(expanded_context, dim = 0)
    assert expanded_context.size() == torch.Size([beam_width, seqlen])
    top_k_ids = top_k_ids.view(beam_width, 1)
    next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)
    assert next_input_ids.size() == torch.Size([beam_width, seqlen+1])
    new_hidden_states, next_logits = model.compute_logits_and_hidden_states(next_input_ids)
    assert new_hidden_states.size() == torch.Size([beam_width, seqlen+1, embed_dim])
    context_hidden = new_hidden_states[:,:seqlen,:]
    assert context_hidden.size() == torch.Size([beam_width, seqlen, embed_dim])
    next_hidden = new_hidden_states[:,seqlen:,:]
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])

    next_id = ranking(context_hidden, next_hidden, top_k_ids, top_k_probs, alpha)       

    next_input_ids = torch.cat([input_ids, next_id], dim = -1)
    assert next_input_ids.size() == torch.Size([1, seqlen+1])
    return next_input_ids


# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    # faith = torch.max(cosine_matrix[:, :prefix_len], dim=-1)[0]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores   # [B*K]
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx


def ContrastiveDecodingOneStepFast(
    model, 
    ids, 
    beam_width, 
    alpha, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    config,
    first_step=False,
    ):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)
    beam_width, var_out = adaptive_k(logit_for_next_step, config)
    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
    # compute new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1), 
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*K, V]
    next_hidden = output.hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]
    
    alpha, var_out_k = adaptive_alpha(logit_for_next_step, beam_width, config)

    
    alpha_list.append(alpha)
    k_list.append(beam_width)
    #kld_uniform_list.append(kld_uniform)
    #kld_cert_list.append(kld_cert)

    var_list.append(var_out)
    var_k_list.append(var_out_k)     
    
    #print(alpha, kld_uniform, kld_cert, val_res, diff2, kld_test)

    selected_idx = ranking_fast(
        context_hidden, 
        next_hidden, 
        top_k_probs,    # [B, K] 
        alpha,
        beam_width,
    )     # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    # next_id: [B, 1]
    
    probab = [F.softmax(element, dim = 0).tolist() for element in logits][0][selected_idx]
    #print(f'Selected idx: {selected_idx}')
    probability_list.append(probab)
    #print(f'probab list length: {len(probability_list)}')
    perplex =  np.exp(np.mean(-np.log(probability_list)))
    perplex_list.append(perplex)
    #print(f'perplex list length: {len(perplex_list)}')

    return next_id, past_key_values, last_hidden_states, logits 


def return_val1():
    return(alpha_list, kld_uniform_list, perplex_list)


def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz*beam_width, num_head, seq_len, esz)    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values