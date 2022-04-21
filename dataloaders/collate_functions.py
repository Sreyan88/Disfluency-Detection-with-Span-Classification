# encoding: utf-8

import torch
from typing import List
import numpy as np
import sys


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
             tokens,type_ids,all_span_idxs_ltoken,morph_idxs, ...
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """

    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    sent_max_len = [x[0].shape[0] for x in batch]
    max_num_span = max(x[3].shape[0] for x in batch)
    output = []

    for field_idx in range(2):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # begin{for the pad_all_span_idxs_ltoken... }
    pad_all_span_idxs_ltoken = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append((0,0))
        pad_all_span_idxs_ltoken.append(sma)
    pad_all_span_idxs_ltoken = torch.Tensor(pad_all_span_idxs_ltoken)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][2]
        pad_all_span_idxs_ltoken[sample_idx, : data.shape[0],:] = data
    output.append(pad_all_span_idxs_ltoken)
    # end{for the pad_all_span_idxs_ltoken... }


    # begin{for the morph feature... morph_idxs}
    pad_morph_len = len(batch[0][3][0])
    pad_morph = [0 for i in range(pad_morph_len)]
    pad_morph_idxs = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append(pad_morph)
        pad_morph_idxs.append(sma)
    pad_morph_idxs = torch.LongTensor(pad_morph_idxs)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_morph_idxs[sample_idx, : data.shape[0], :] = data
    output.append(pad_morph_idxs)
    # end{for the morph feature... morph_idxs}


    for field_idx in [4,5,6,7]:
        pad_output = torch.full([batch_size, max_num_span], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    words = []
    for sample_idx in range(batch_size):
        words.append(batch[sample_idx][8])
    output.append(words)


    all_span_word = []
    for sample_idx in range(batch_size):
        all_span_word.append(batch[sample_idx][9])
    output.append(all_span_word)

    all_span_idxs = []
    for sample_idx in range(batch_size):
        all_span_idxs.append(batch[sample_idx][10])
    output.append(all_span_idxs)

    # all_span_heads = []
    # for sample_idx in range(batch_size):
    #     all_span_heads.append(batch[sample_idx][11])

    adjs = [head_to_adj(max_length, batch[sample_idx][11], batch[sample_idx][12],sent_max_len[sample_idx]) for sample_idx in range(batch_size)]
     
    output.append(adjs)

    return output



def head_to_adj(max_len, heads, orig_to_tok_index, sent_max_len):
        """
        Convert a tree object to an (numpy) adjacency matrix.
        """
        directed = 0
        self_loop = False #config.adj_self_loop
        ret = np.zeros((max_len, max_len), dtype=np.float32)
        
            
        # for i, head in enumerate(heads):
        #     if head == -1:
        #         continue
            
        #     ret[head, orig_to_tok_index[i]] = 1

        i = 0
        head = -1
        while i < sent_max_len:
            if i in orig_to_tok_index:
                head += 1
                ret[int(heads[head]), i] = 1
            else:
                ret[int(heads[head]), i] = 1
            i += 1
            
        if not directed:
            ret = ret + ret.T

        # if self_loop:
        #     for i in range(len(inst.input.words)):
        #         ret[i, i] = 1

        return ret

# def head_to_adj_label(max_len, heads, dep_label_ids):
#     """
#     Convert a tree object to an (numpy) adjacency matrix.
#     """
#     directed = 0
#     # self_loop = config.adj_self_loop

#     dep_label_ret = np.zeros((max_len, max_len), dtype=np.long)

#     for i, head in enumerate(heads):
#         if head == -1:
#             continue
#         dep_label_ret[head, i] = dep_label_ids[i]

#     if not directed:
#         dep_label_ret = dep_label_ret + dep_label_ret.T

#     # if self_loop:
#     #     for i in range(len(inst.input.words)):
#     #         dep_label_ret[i, i] = config.root_dep_label_id

#     return dep_label_ret
