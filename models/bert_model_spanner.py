# encoding: utf-8


import torch
import torch.nn as nn
import sys
from transformers import BertModel, BertPreTrainedModel,RobertaModel
import math

from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F

class BertNER(BertPreTrainedModel):
    def __init__(self, config,args):
        super(BertNER, self).__init__(config)
        self.bert = BertModel(config)
        config.hidden_size = config.hidden_size * 2
        self.args = args
        if 'roberta' in self.args.bert_config_dir:
            self.bert = RobertaModel(config)
            print('use the roberta pre-trained model...')


        # self.start_outputs = nn.Linear(config.hidden_size, 2)
        # self.end_outputs = nn.Linear(config.hidden_size, 2)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        # self.span_embedding = SingleLinearClassifier(config.hidden_size * 2, 1)

        self.hidden_size = config.hidden_size

        self.span_combination_mode = self.args.span_combination_mode
        self.max_span_width = args.max_spanLen
        self.n_class = args.n_class
        self.tokenLen_emb_dim = self.args.tokenLen_emb_dim # must set, when set a value to the max_span_width.

        # if self.args.use_tokenLen:
        #     self.tokenLen_emb_dim = self.args.tokenLen_emb_dim
        # else:
        #     self.tokenLen_emb_dim = None




        print("self.max_span_width: ", self.max_span_width)
        print("self.tokenLen_emb_dim: ", self.tokenLen_emb_dim)

        #  bucket_widths: Whether to bucket the span widths into log-space buckets. If `False`, the raw span widths are used.

        self._endpoint_span_extractor = EndpointSpanExtractor(config.hidden_size,
                                                              combination=self.span_combination_mode,
                                                              num_width_embeddings=self.max_span_width,
                                                              span_width_embedding_dim=self.tokenLen_emb_dim,
                                                              bucket_widths=True)


        self.linear = nn.Linear(10, 1)
        self.score_func = nn.Softmax(dim=-1)

        # import span-length embedding
        self.spanLen_emb_dim =args.spanLen_emb_dim
        self.morph_emb_dim = args.morph_emb_dim
        input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim
        if self.args.use_spanLen and not self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim+self.spanLen_emb_dim
        elif not self.args.use_spanLen and self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.morph_emb_dim
        elif  self.args.use_spanLen and self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.spanLen_emb_dim + self.morph_emb_dim


        self.span_embedding = MultiNonLinearClassifier(input_dim, self.n_class,
                                                       config.model_dropout)

        self.spanLen_embedding = nn.Embedding(args.max_spanLen+1, self.spanLen_emb_dim, padding_idx=0)

        self.morph_embedding = nn.Embedding(len(args.morph2idx_list) + 1, self.morph_emb_dim, padding_idx=0)
        self.W = nn.ModuleList()
        for layer in range(2):
            self.W.append(nn.Linear(768, 768)).to('cuda')

        self.gate = nn.Linear(1536, 768)

        # self.lstm_f = MyLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to('cuda')
        # self.lstm_b = MyLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to('cuda')

    def forward(self,loadall, all_span_lens, all_span_idxs_ltoken, input_ids, adjs, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
            all_span_idxs: the span-idxs on token-level. (bs, n_span)
            pos_span_mask: 0 for negative span, 1 for the positive span. SHAPE: (bs, n_span)
            pad_span_mask: 1 for real span, 0 for padding SHAPE: (bs, n_span)
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # bert_outputs = torch.cat((bert_outputs, adjs), 2)
        
        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        # print(sequence_heatmap.shape)
        adjs = torch.tensor(adjs, device='cuda')
        denom = adjs.sum(2).unsqueeze(2) + 1
        graph_output = None
        for l in range(2):
            Ax = adjs.bmm(sequence_heatmap)  ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)   ## N x m
            AxW = AxW + self.W[l](sequence_heatmap)  ## self loop  N x h
            AxW = AxW / denom
            graph_output = torch.relu(AxW)

        # gate
        merge_bert_graph = torch.cat((sequence_heatmap, graph_output), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_bert_graph)) 
        gated_converted = torch.mul(gate_value, graph_output)
        sequence_heatmap = torch.cat((sequence_heatmap, gated_converted), 2)

        # print(sequence_heatmap.shape)

        # lstm_out = self.lstm_f(sequence_heatmap, graph_output)
        # # backward LSTM
        # word_rep_b = masked_flip(sequence_heatmap, word_seq_len.tolist())
        # c_b = masked_flip(graph_input, word_seq_len.tolist())
        # lstm_out_b = self.lstm_b(word_rep_b, c_b)
        # lstm_out_b = masked_flip(lstm_out_b, word_seq_len.tolist())

        # feature_out = torch.cat((lstm_out, lstm_out_b), dim=2)
        # feature_out = self.drop_lstm(feature_out)

        all_span_rep = self._endpoint_span_extractor(sequence_heatmap, all_span_idxs_ltoken.long()) # [batch, n_span, hidden]
        if not self.args.use_spanLen and not self.args.use_morph:
            # roberta_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            # sequence_heatmap = roberta_outputs[0]  # [batch, seq_len, hidden]
            #
            # # get span_representation with different labels.
            # # put the positive span in the first and use the span_mask to keep the positive span.
            # # then, for the negative span, we can random sample n_pos_span *2
            # all_span_rep = self._endpoint_span_extractor(sequence_heatmap, all_span_idxs_ltoken.long())
            all_span_rep = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)

        elif self.args.use_spanLen and not self.args.use_morph:
            spanlen_rep = self.spanLen_embedding(all_span_lens) # (bs, n_span, len_dim)
            spanlen_rep = F.relu(spanlen_rep)
            all_span_rep = torch.cat((all_span_rep, spanlen_rep), dim=-1)
            all_span_rep = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)
        elif not self.args.use_spanLen and self.args.use_morph:
            morph_idxs = loadall[3]
            span_morph_rep = self.morph_embedding(morph_idxs) #(bs, n_span, max_spanLen, dim)
            span_morph_rep = torch.sum(span_morph_rep, dim=2) #(bs, n_span, dim)

            all_span_rep = torch.cat((all_span_rep, span_morph_rep), dim=-1)
            all_span_rep = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)

        elif self.args.use_spanLen and self.args.use_morph:
            morph_idxs = loadall[3]
            span_morph_rep = self.morph_embedding(morph_idxs) #(bs, n_span, max_spanLen, dim)
            span_morph_rep = torch.sum(span_morph_rep, dim=2) #(bs, n_span, dim)

            spanlen_rep = self.spanLen_embedding(all_span_lens)  # (bs, n_span, len_dim)
            spanlen_rep = F.relu(spanlen_rep)

            all_span_rep = torch.cat((all_span_rep,spanlen_rep, span_morph_rep), dim=-1)
            all_span_rep = self.span_embedding(all_span_rep)  # (batch,n_span,n_class)


        return all_span_rep
class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1),  self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz),  self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):

        # # # new standard lstm
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)


        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii,uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x) 

        return ht, Ct_x, Ct_m 

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m= self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2) ##batch_size x max_len x hidden
        return hidden_seq

def masked_flip(padded_sequence: torch.Tensor, sequence_lengths):
    """
        Flips a padded tensor along the time dimension without affecting masked entries.
        # Parameters
        padded_sequence : `torch.Tensor`
            The tensor to flip along the time dimension.
            Assumed to be of dimensions (batch size, num timesteps, ...)
        sequence_lengths : `torch.Tensor`
            A list containing the lengths of each unpadded sequence in the batch.
        # Returns
        A `torch.Tensor` of the same shape as padded_sequence.
        """
    assert padded_sequence.size(0) == len(
        sequence_lengths
    ), f"sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}"
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [
        flipped_padded_sequence[i, num_timesteps - length :]
        for i, length in enumerate(sequence_lengths)
    ]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
