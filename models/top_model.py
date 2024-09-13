
import torch
import torch.nn as nn
import numpy as np
from models.basic_models import *
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class IMAD_top(nn.Module):
    def __init__(
            self, pretrain, use_func, n_layers, n_head, d_k, d_v,
            d_model, d_inner, other_feature_dim, iffnn_hidden_dimensions, act_func, use_batnorm,
            model_name='star_transformer', dropout=0.1, use_dropout=False, norm_type='batch', use_last_norm=False,
            use_last_param=False):
        print("malware_detector_top")

        super().__init__()
        self.model_name = model_name
        self.pretrain = pretrain
        self.use_func = use_func
        self.use_batnorm = use_batnorm
        self.use_last_norm = use_last_norm
        self.use_last_param = use_last_param
        if use_func:
            self.binary_level = StarPlusTransformer(num_layers=n_layers,
                                                    num_head=n_head, head_dim=d_k,
                                                    hidden_size=d_model, d_inner=d_inner, dropout=0.1)
        else:
            d_model = 0
        if pretrain:
            other_feature_dim = 0
        assert(not (pretrain and (not use_func)))
        previous_dim = d_model + other_feature_dim
        if norm_type == 'layer':
            print("last layer layer norm")
            self.last_norm = nn.LayerNorm(previous_dim)
        else:
            self.last_norm = nn.BatchNorm1d(previous_dim)
        dic = OrderedDict()
        for i, dim in enumerate(iffnn_hidden_dimensions):

            if use_dropout:
                dic['drop' + str(i)] = nn.Dropout(p=dropout)
            if use_batnorm:
                if norm_type == 'layer':
                    dic['batchnorm' + str(i)] = nn.LayerNorm(previous_dim)
                else:
                    dic['batchnorm' + str(i)] = nn.BatchNorm1d(previous_dim)
            lay = nn.Linear(previous_dim, dim)
            previous_dim = dim
            dic['linear' + str(i)] = lay
            if act_func == 'tanh':
                dic['act_func' + str(i)] = nn.Tanh()
            else:
                assert (act_func == 'relu')
                dic['act_func' + str(i)] = nn.ReLU()

        n_hid = len(iffnn_hidden_dimensions)
        if use_dropout:
            dic['drop' + str(n_hid)] = nn.Dropout(p=dropout)
        if use_batnorm:
            dic['batchnorm' + str(n_hid)] = nn.BatchNorm1d(previous_dim)
        lay = nn.Linear(previous_dim, d_model + other_feature_dim)
        dic['linear' + str(n_hid)] = lay

        # dic['act_func'+str(n_hid)]=nn.Tanh()

        self.iffnnpart1 = nn.Sequential(dic)

        # self.last_linear = nn.Linear(1,1)
        self.last_weight = torch.nn.Parameter(torch.rand([d_model + other_feature_dim]))
        self.register_parameter(name='weight', param=self.last_weight)
        self.last_bias = torch.nn.Parameter(torch.zeros([1]))
        self.register_parameter(name='bias', param=self.last_bias)


    def compute_feature_weight(self, full_features):
        if not type(full_features) == torch.Tensor:
            full_features = torch.tensor(full_features)
        out = full_features
        out = self.iffnnpart1(out)
        if self.use_last_param:
            out = out * self.last_weight
        if self.use_last_norm:
            full_features = self.last_norm(full_features)
        out = full_features * out
        return out

    def forward1(self, full_features):
        # pad ones use 0 for now. It doesn't matter
        out = self.compute_feature_weight(full_features)
        out = out.sum(dim=1) + self.last_bias
        out = torch.sigmoid(out)
        return out

    def forward(self, other_features, src_seq1=None, func1_mask=None, with_relevance=False):
        if with_relevance:
            return self.forward_with_relevance(other_features, src_seq1, func1_mask)
        # pad ones use 0 for now. It doesn't matter
        if self.use_func:
            if self.model_name == 'star_transformer':
                _, func_reps = self.binary_level(src_seq1, func1_mask)
                bin_functions = func_reps
            else:
                func_reps = self.binary_level(src_seq1, func1_mask, 0)
                bin_functions = func_reps[:, 0, :]
            if self.pretrain:
                full_features = bin_functions
            else:
                full_features = torch.cat([bin_functions, other_features], dim=1)
        else:
            full_features = other_features
        print("full_features.shape:",full_features.shape)
        out = self.forward1(full_features)
        return out

    def forward_with_relevance(self, other_features, src_seq1=None, func1_mask=None):
        # pad ones use 0 for now. It doesn't matter
        if self.use_func:
            if self.model_name == 'star_transformer':
                _, func_reps = self.binary_level(src_seq1, func1_mask)
                bin_functions = func_reps
            else:
                func_reps = self.binary_level(src_seq1, func1_mask, 0)
                bin_functions = func_reps[:, 0, :]
            full_features = torch.cat([bin_functions, other_features], dim=1)
        else:
            full_features = other_features
        if not type(full_features) == torch.Tensor:
            full_features = torch.tensor(full_features)
        inp = full_features
        out = full_features
        out = self.iffnnpart1(out)
        if self.use_last_param:
            out = out * self.last_weight
        if self.use_last_norm:
            full_features = self.last_norm(full_features)
        relevance = relevance = torch.stack([-out, out], dim=1).permute(0, 2, 1).contiguous()
        out = out.sum(dim=1) + self.last_bias
        out = torch.sigmoid(out)
        return inp, out, relevance

    def attribute(self, other_features, device, n_head, src_seq1=None, return_attns=True, no_att_dropout=True):

        if self.use_func:
            src_seq1 = torch.tensor(src_seq1, dtype=torch.float32).to(device)
            other_features = torch.tensor(other_features).to(device)
            func1_mask = torch.ones(src_seq1.shape[:2], dtype=torch.long)
            n_sam = len(src_seq1)
            n_func = src_seq1.shape[1]
            if self.model_name == 'star_transformer':
                _, func_reps, enc_slf_attn_list = self.binary_level(src_seq1, func1_mask, device, return_attns)
                bin_functions = func_reps
                last_layer_att = enc_slf_attn_list
                assert len(last_layer_att) == n_sam
                attended_funcs = []
                for att_persam in last_layer_att:
                    at = []
                    head = att_persam
                    assert len(head) == n_func
                    head = list(head)
                    head = [(i, f) for i, f in enumerate(head)]
                    head = sorted(head, key=operator.itemgetter(1), reverse=True)
                    for i in range(len(head)):
                        at.append((head[i][0], head[i][1]))
                    attended_funcs.append(at)
            else:
                func_reps, enc_slf_attn_list = self.binary_level(src_seq1, func1_mask, 0, device, return_attns,
                                                                 no_att_dropout)
                bin_functions = func_reps[:, 0, :]

                last_layer_att = enc_slf_attn_list[-1].view(n_head, n_sam, n_func + 1, n_func + 1).permute(1, 0, 2,
                                                                                                           3).data.cpu().numpy()
                last_layer_att = last_layer_att[:, :, 0, :]

                assert len(last_layer_att) == n_sam
                attended_funcs = []
                for att_persam in last_layer_att:
                    at = []
                    for head in att_persam:
                        assert len(head) == n_func + 1
                        head = list(head)
                        sca = 1 - head[0]
                        head = [(i, f) for i, f in enumerate(head[1:])]
                        head = sorted(head, key=operator.itemgetter(1), reverse=True)
                        for i in range(min(2, len(head))):
                            at.append((head[i][0], head[i][1] / sca))
                    attended_funcs.append(at)

            full_features = torch.cat([bin_functions, other_features], dim=1)
        else:
            full_features = other_features
        att = self.compute_feature_weight(full_features)
        out = att.sum(dim=1)
        out = torch.sigmoid(out)
        out = out.view(out.shape[0], )
        pred = np.round(out.data.cpu().numpy()).astype(int)
        att_np = att.data.cpu().numpy()
        att_result = []
        n_dim_func = bin_functions.shape[1]
        # print('n_dim_func:',n_dim_func)
        for f_l in att_np:
            lis = [(i, f) for i, f in enumerate(f_l)]
            func_cont = 0.
            for i in range(n_dim_func):
                func_cont += lis[i][1]
            lis = lis[n_dim_func:]
            # print('func_cont:',func_cont)
            lis.append((0, func_cont))
            lis = sorted(lis, key=operator.itemgetter(1), reverse=True)
            att_result.append(lis)
        if use_func:
            return out.data.cpu().numpy(), pred, att_result, attended_funcs
        else:
            return out.data.cpu().numpy(), pred, att_result