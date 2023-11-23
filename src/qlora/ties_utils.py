import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1).cpu() for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the reference dict
    torch.nn.utils.vector_to_parameters(vector.cpu(), sorted_reference_dict.values())
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return (M * final_mask).squeeze(), final_mask
    return (M * final_mask).squeeze()


def compress_and_update_llama_adaptors(model, k, replace_factor):
    if k == 100:
        return

    weights = get_peft_model_state_dict(model)
    remove_keys = []
    flat_ptm = 0
    flat_weights = state_dict_to_vector(weights, remove_keys=remove_keys)
    tv = flat_weights - flat_ptm
    mean, std = tv.mean(), tv.std()
    topk_flat_tv, topk_mask = topk_values_mask(tv, K=k, return_mask=True)
    assert tv[~topk_mask].abs().max() <= topk_flat_tv[topk_mask].abs().min()
    if replace_factor == -2:
        print("Using STC compression!")
        alpha = topk_flat_tv[topk_flat_tv != 0].mean()
        updated_flat_tv = alpha * topk_flat_tv.sign()
    elif replace_factor > 0:
        alpha = replace_factor * std
        updated_flat_tv = alpha * topk_flat_tv.sign()
    elif replace_factor == -1:
        print("Using Just Pruning!")
        updated_flat_tv = topk_flat_tv

    updated_flat_weights = updated_flat_tv + flat_ptm
    updated_weights = vector_to_state_dict(
        updated_flat_weights, weights, remove_keys=remove_keys
    )
    missing_keys, unexpected_keys = set_peft_model_state_dict(model, updated_weights)
    assert set(missing_keys) - set(weights.keys()) == set(missing_keys)
    assert len(unexpected_keys) == 0


def get_compressed_lora_weights(state_dict, k, replace_factor):
    if k == 100:
        return

    weights = copy.deepcopy(state_dict)
    remove_keys = []
    flat_ptm = 0
    flat_weights = state_dict_to_vector(weights, remove_keys=remove_keys)
    tv = flat_weights - flat_ptm
    mean, std = tv.mean(), tv.std()
    topk_flat_tv, topk_mask = topk_values_mask(tv, K=k, return_mask=True)
    assert tv[~topk_mask].abs().max() <= topk_flat_tv[topk_mask].abs().min()

    if replace_factor == -2:
        print("Using STC compression!")
        alpha = topk_flat_tv[topk_flat_tv != 0].mean()
        updated_flat_tv = alpha * topk_flat_tv.sign()
    elif replace_factor > 0:
        alpha = replace_factor * std
        updated_flat_tv = alpha * topk_flat_tv.sign()
    elif replace_factor == -1:
        print("Using Just Pruning!")
        updated_flat_tv = topk_flat_tv

    updated_flat_weights = updated_flat_tv + flat_ptm
    updated_weights = vector_to_state_dict(
        updated_flat_weights, weights, remove_keys=remove_keys
    )
    return updated_weights
