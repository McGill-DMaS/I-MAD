''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from keras_preprocessing import sequence

pad_sequences = sequence.pad_sequences

'''
PAD = 0    # not used
UNK = 1
BOS = 2    #[BOS,EMPTY_OPRA,EMPTY_OPRA]     not used
MEM = 3
VAL = 4
EMPTY_OPRA = 5
MASK = 6    #[MASK,EMPTY_OPRA,EMPTY_OPRA]  EMPTY inst   also used to indicate empty basic block
TARGET = 7    #[TARGET,EMPTY_OPRA,EMPTY_OPRA]  target inst to predict
'''
PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def pad(x,value,maxi=0):
    if maxi > 0:
        with_maxi = True
    else:
        with_maxi = False
    masks = []
    for i,sample in enumerate(x):
        length = len(sample)
        if with_maxi and length > maxi:
            x = x[:maxi]
        masks.append(length)
        if length > maxi and not with_maxi:
            maxi = length
    if type(x) == list:
        return pad_sequences(x, maxlen=maxi, padding='post', value=[value]),torch.tensor(masks)
    else:
        return pad_sequences(x.cpu(), maxlen=maxi, padding='post', value=[value]),torch.tensor(masks)


def get_attn_key_pad_mask_full_inst_masked_is_zero_two_dimension(seq_k, seq_q,pad):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    seq_k_opcode = seq_k[:,:,0]
    padding_mask = seq_k_opcode.ne(pad)  #problem is here
    return padding_mask

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_1(output)
        output = F.relu(output)
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class  mask_star_satelitte_planet_pretraining_model(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_hidden, n_out,token_map, dropout=0.2):
        
        super().__init__()

        self.token_map = token_map


        #self.token_map = {'PAD': 0, 'OOV': 1, 'BOS': 2, 'MEM': 3, 'LOC': 4, 'COMMA': 5, 'PTR': 6, 'RVA': 7, 'OFFSETSUB': 8,
        #         'OFFSET': 9, 'SUB': 10, 'WORD': 11, 'VAL': 12, 'EMPTY_OPRA': 13, 'MASK': 14, 'TARGET': 15, 'UNK': 16}

        self.encoder = AssemblyCodeEncoderStarPlusTransformer(n_src_vocab, len_max_seq, d_word_vec,
                                                              n_layers, n_head, d_k, d_v,
                                                              d_model, d_inner, token_map, dropout=dropout)

        self.fc1 = nn.Linear(6*d_model, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self.fcopr1_1 = nn.Linear(6*d_model, n_hidden)
        self.fcopr1_2 = nn.Linear(n_hidden, n_out)
    
    
        self.fcopr2_1 = nn.Linear(6*d_model, n_hidden)
        self.fcopr2_2 = nn.Linear(n_hidden, n_out)
    
    def forward(self, src_seq,  positions, device, return_attns=False):      # now opcode and operand mixed
        # src_seq and positions are just lists sec_seq is not padded

        nodes, relay = self.encoder(src_seq, device)
        target0 = relay
        
        enc_output = nodes    #very important, because the positions are for the original without padding
        
        target = enc_output[list(range(len(positions))),positions,:]
        final_target = torch.cat([target0,target],dim=1)
        
        opcode_out = F.relu(self.fc1(final_target))
        opcode_out = self.fc2(opcode_out)
        m, n = opcode_out.shape
        
        oprand1_out = F.relu(self.fcopr1_1(final_target))
        oprand1_out = self.fcopr1_2(oprand1_out)
        
        
        oprand2_out = F.relu(self.fcopr2_1(final_target))
        oprand2_out = self.fcopr2_2(oprand2_out)

        final_out = torch.cat([opcode_out,oprand1_out,oprand2_out],dim=1).view(3*m,n)
        
        return final_out

# Updated from Start-Transformer from https://github.com/fastnlp/fastNLP/
class AssemblyCodeEncoderStarPlusTransformer(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, token_map, dropout=0.1):

        super().__init__()

        self.token_map = token_map

        n_position = len_max_seq
        max_len = len_max_seq
        hidden_size = 3*d_model
        num_layers = n_layers
        num_head = n_head
        head_dim = d_k
        self.iters = num_layers
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec)
        
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.iters)])
        self.pwff = nn.ModuleList([PositionwiseFeedForward(hidden_size,d_inner) for _ in range(self.iters)])
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])
        
        if max_len is not None:
            self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, 3*d_word_vec, padding_idx=0),
            freeze=True)
        else:
            self.pos_emb = None
    
    def forward(self, data, device):
        src_seq = data
        
        src_seq,_ = pad(src_seq,[self.token_map['MASK'],self.token_map['EMPTY_OPRA'],self.token_map['EMPTY_OPRA']])
        
        src_seq = Variable(torch.LongTensor(np.array(src_seq))).to(device)
        
        
        mask = get_attn_key_pad_mask_full_inst_masked_is_zero_two_dimension(seq_k=src_seq, seq_q=src_seq, pad = self.token_map['MASK'])
        
        emb_tmp = self.src_word_emb(src_seq)
        data = emb_tmp.view(emb_tmp.size()[0],emb_tmp.size()[1],-1)
        
        
        def norm_func_o(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        def norm_func(f, x,B, L, H):
            # B, H, L, 1
            tmp1 = x.permute(0,2,1,3).view(B, L, H)
            tmp2 = f(tmp1)
            tmp3 = tmp2.view(B, L, H,1).permute(0,2,1,3)
            return tmp3
        
        B, L, H = data.size()
        mask = (mask.eq(False))  # flip the mask for masked_fill_
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)
        
        embs = data.permute(0, 2, 1)[:, :, :, None]  # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device) \
                             .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P
        embs = norm_func_o(self.emb_drop, embs)
        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        for i in range(self.iters):
            ax = relay.expand(B, H, 1, L)
            nodes = F.leaky_relu(norm_func_o(self.norm[i],self.ring_att[i](norm_func(self.pwff[i], nodes,B, L, H), ax=ax)))
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))
            
            nodes = nodes.masked_fill_(ex_mask, 0)
        
        nodes = nodes.view(B, H, L).permute(0, 2, 1)
        
        return nodes, relay.view(B, H)


# Updated from Start-Transformer from https://github.com/fastnlp/fastNLP/
class StarPlusTransformer(nn.Module):

    def __init__(self, hidden_size, num_layers, num_head, head_dim, d_inner, dropout=0.1, max_len=None):
        super().__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.iters)])
        self.pwff = nn.ModuleList([PositionwiseFeedForward(hidden_size, d_inner) for _ in range(self.iters)])
        # self.emb_fc = nn.Conv2d(hidden_size, hidden_size, 1)
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])

        if max_len is not None:
            self.pos_emb = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None

    def forward(self, data, mask, with_att=False):

        def norm_func_o(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        def norm_func(f, x, B, L, H):
            # B, H, L, 1
            tmp1 = x.permute(0, 2, 1, 3).view(B, L, H)
            tmp2 = f(tmp1)
            tmp3 = tmp2.view(B, L, H, 1).permute(0, 2, 1, 3)
            return tmp3

        B, L, H = data.size()
        mask = (mask.eq(False))  # flip the mask for masked_fill_
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 2, 1)[:, :, :, None]  # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device) \
                             .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P
        embs = norm_func_o(self.emb_drop, embs)
        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        for i in range(self.iters):
            ax = relay.expand(B, H, 1, L)
            nodes = F.leaky_relu(
                norm_func_o(self.norm[i], self.ring_att[i](norm_func(self.pwff[i], nodes, B, L, H), ax=ax)))

            if with_att:
                relay_, att = self.star_att[i](relay, torch.cat([relay, nodes], 2), smask, with_att)
                relay = F.leaky_relu(relay_)
            else:
                relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask, with_att))

            nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)
        if with_att:
            return nodes, relay.view(B, H), att
        else:
            return nodes, relay.view(B, H)



class _MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(_MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / np.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret


class _MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(_MSA2, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None, with_att=False):
        # x: B, H, 1, 1, 1 y: B H 1 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / np.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        if with_att:
            return self.WO(att), F.softmax(alphas.sum(dim=1)[:, 0, 1:], dim=1)
        else:
            return self.WO(att)
