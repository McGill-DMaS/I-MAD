import os


from torch.autograd import Variable
from models.basic_models import *


class hierarchical_transformer(nn.Module):
    def __init__(
            self, token_map,
            n_src_vocab, len_max_seq, d_word_vec,
            block_n_layers, function_n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_hidden, top_layer='cos', dropout=0.1, model_name='star_transformer'):

        super().__init__()

        self.token_map = token_map
        self.model_name = model_name

        print('using start_transformer')
        self.block_level = AssemblyCodeEncoderStarPlusTransformer(n_src_vocab, len_max_seq, d_word_vec,
                                                                  block_n_layers, n_head, d_k, d_v,
                                                                  d_model, d_inner, token_map, dropout=0.1)
        self.function_level = StarPlusTransformer(num_layers=function_n_layers,
                                                  num_head=n_head, head_dim=d_k,
                                                  hidden_size=3 * d_model, d_inner=d_inner, dropout=0.1)
        self.empty_block_pad = 0

        self.d_model = d_model
        self.top_layer = top_layer
        if top_layer == 'feed':
            self.fc1 = nn.Linear(2 * d_model, n_hidden)
            self.fc2 = nn.Linear(n_hidden, 1)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def pad_hierarchical_data_pairs(self, func1, func2,device):
        n_sample = len(func1)
        longest_block = 0
        longest_function = 0
        for t in [func1, func2]:
            for func in t:
                if len(func) > longest_function:
                    longest_function = len(func)
                for block in func:
                    if len(block) > longest_block:
                        longest_block = len(block)
        # print('length of blocks: {} length of functions: {}'.format(longest_block, longest_function))
        function1_mask = []
        function2_mask = []
        for t, mask in [(func1, function1_mask), (func2, function2_mask)]:
            for func in t:
                tmp = []
                mask.append([1] * len(func) + [self.empty_block_pad] * (longest_function - len(func)))
                for block in func:
                    block.extend(
                        [[self.token_map['MASK'], self.token_map['EMPTY_OPRA'], self.token_map['EMPTY_OPRA']]] * (
                                    longest_block - len(block)))
                func.extend([[[self.token_map['MASK'], self.token_map['EMPTY_OPRA'],
                               self.token_map['EMPTY_OPRA']]] * longest_block for i in
                             range(longest_function - len(func))])
        func1 = torch.tensor(func1).reshape(n_sample * longest_function, longest_block, 3).to(device)
        func2 = torch.tensor(func2).reshape(n_sample * longest_function, longest_block, 3).to(device)
        function1_mask = torch.tensor(function1_mask).to(device)
        function2_mask = torch.tensor(function2_mask).to(device)
        return func1, func2, function1_mask, function2_mask, longest_block, longest_function

    def forward(self, src_seq1, src_seq2, device, return_attns=False):
        # src_seq1, src_seq2 are not padded. they are just list of blocks that have different number of instructions

        src_seq1, src_seq2, func1_mask, func2_mask, longest_block, longest_function = self.pad_hierarchical_data_pairs(
            src_seq1, src_seq2,device)

        if self.model_name == 'star_transformer':
            nodes, relay = self.block_level(src_seq1, device)
            enc_output1 = relay
            nodes, relay = self.block_level(src_seq2, device)
            enc_output2 = relay
        else:
            # be careful, the position should be the REAL POSITION + 1, since there is a CLS at the beginning
            enc_output1 = self.block_level(src_seq1, device)[:, 0, :]
            enc_output2 = self.block_level(src_seq2, device)[:, 0, :]
        # print('enc_output1',enc_output1.size())
        # print('longest_function,self.d_model',longest_function,self.d_model)
        block1_representation = enc_output1.view(-1, longest_function, 3 * self.d_model)
        block2_representation = enc_output2.view(-1, longest_function, 3 * self.d_model)

        # print('block1_representation',block1_representation.size(),block1_representation)

        nodes, relay = self.function_level(block1_representation, func1_mask)
        func1_representation = relay
        nodes, relay = self.function_level(block2_representation, func2_mask)
        func2_representation = relay

        if self.top_layer == 'feed':
            conca1 = torch.cat([func1_representation, func2_representation], dim=1)
            conca2 = torch.cat([func2_representation, func1_representation], dim=1)
            out = F.relu(self.fc1(conca1) + self.fc1(conca2))
            out = torch.sigmoid(self.fc2(out))
        else:
            out = self.cos(func1_representation, func2_representation)
        return out


    def calc_similarity(self, rep1, rep2):
        if self.top_layer == 'feed':
            conca1 = torch.cat([rep1, rep2], dim=1)
            conca2 = torch.cat([rep2, rep1], dim=1)
            out = F.relu(self.fc1(conca1) + self.fc1(conca2))
            out = torch.sigmoid(self.fc2(out))
        else:
            out = self.cos(rep1, rep2)

        return out


