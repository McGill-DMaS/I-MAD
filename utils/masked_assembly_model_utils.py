import os

import operator

from utils.utils import *
import _pickle as cPickle

import json

import progressbar

import random

from collections import Counter

'''

LOC:  loc_
"jnz", "loc_100011D1"

comma:  : in operand
"call", "ds:GlobalAlloc"]
"movsx", "ebx", "ds:byte_100068A0[ecx]"

PTR:
ecx", "byte ptr [eax+ebx]"
"dword ptr [edx]", "6"

RVA: startswith rva
"movzx", "ecx", "rva byte_642FFA8225C[r13+rax*8]"]

OFFSETSUB:
"dword ptr [edi+48h]", "offset sub_DF17F0"]

OFFSET：
"push", "offset dword_E09308"]


SUB:
"call", "sub_DF19C0"

WORD:
"mov", "dx", "word_E07078[ebx]"
'''


def get_mask_data_from_digit_blocks(blocks, token_map):
    inputs = []
    masks = []
    outputs = []

    for block in blocks:
        ind = random.randint(0, len(block) - 1)
        out = []
        for token in block[ind]:
            out.append(token)
        out.extend([token_map['EMPTY_OPRA']] * (3 - len(block[ind])))
        outputs.append(out)  # Be Careful
        masks.append(ind)
        block[ind] = [token_map['TARGET'], token_map['EMPTY_OPRA'], token_map['EMPTY_OPRA']]
        for i in range(len(block)):
            block[i] = block[i] + [token_map['EMPTY_OPRA']] * (3 - len(block[i]))
        inputs.append(block)
    return inputs, masks, outputs




def get_digit_basic_block_dataset(maximum_length=50, tokenmap_size=2000, paths=[]):
    f_name = 'block_dataset' + str(maximum_length) + '.pkl'
    if os.path.exists(f_name):
        f = open(f_name, 'rb')
        digit_basic_blocks = cPickle.load(f)
        token_map = cPickle.load(f)
        f.close()
        print('digit_basic_block_dataset loaded')
        return digit_basic_blocks, token_map
    print('digit_basic_block_dataset not loaded')
    operand_count = Counter()
    opcode_count = Counter()
    all_blocks = []
    for path in paths:
        files = get_recursive_file_list(path)
        bar = progressbar.ProgressBar(maxval=len(files), prefix='code from' + path).start()
        i = 1
        for f in files:
            if i > 1000:
                break
            bar.update(i)
            if f.endswith('.json') or f.endswith('.tagged'):
                try:
                    funcs = read_assembly_functions_from_json(f)
                except Exception as e:
                    print(e)
                    continue
                i += 1
                for func in funcs:
                    blocks = func['blocks']
                    for block in blocks:
                        src = block['src']
                        all_blocks.append(src)
                        for inst in src:
                            opcode_count[inst[1]] += 1
                            if len(inst) > 2:
                                for j in range(2, len(inst)):
                                    operand_count[inst[j]] += 1

    # Don't change. If change, also need to change AssemblyCodeEncoder(nn.Module) in transformer.py and other places
    token_map = {'PAD': 0, 'OOV': 1, 'BOS': 2, 'MEM': 3, 'LOC': 4, 'COMMA': 5, 'PTR': 6, 'RVA': 7, 'OFFSETSUB': 8,
                 'OFFSET': 9, 'SUB': 10, 'WORD': 11, 'VAL': 12, 'EMPTY_OPRA': 13, 'MASK': 14, 'TARGET': 15, 'UNK': 16}
    ind = len(token_map)


    for k in opcode_count.keys():
        assert k not in token_map
        token_map[k] = ind
        ind += 1
    print('opcode_count size:', len(opcode_count))


    sorted_operand_count = sorted(operand_count.items(), key=operator.itemgetter(1), reverse=True)

    for k in sorted_operand_count:
        if k in token_map:
            print('k in opcode_count:', k in opcode_count)
            print(k)
            continue
        # assert k not in token_map
        token_map[k] = ind
        ind += 1
        if len(token_map) >= tokenmap_size:
            break

    print('operand_count size:', len(operand_count))
    print('token map size:', len(token_map))

    r_token_map = {}
    for k, v in token_map.items():
        r_token_map[v] = k

    leng = len(r_token_map)
    exceed = []
    if leng in r_token_map:
        i = 0
        while True:
            if leng + i in r_token_map:
                exceed.append(r_token_map[leng + i])
                i += 1
            else:
                break
    print('exceed:', len(exceed), exceed)
    empty = []
    for i in range(leng):
        if i not in r_token_map:
            empty.append(i)
    print('empty:', len(empty), empty)
    assert len(empty) == len(exceed)
    for k, v in zip(exceed, empty):
        token_map[k] = v

    r_token_map = {}
    for k, v in token_map.items():
        r_token_map[v] = k
    assert leng not in r_token_map
    print('r_token_map size:', len(r_token_map))
    digit_basic_blocks = []
    bar = progressbar.ProgressBar(maxval=len(all_blocks), prefix='digit_basic_blocks').start()


    i = 0
    for block in all_blocks:
        bar.update(i)
        i += 1
        if len(block) > maximum_length:
            continue
        new_digit_block = []
        for inst in block:
            d_inst = []
            for t in inst[1:4]:
                if t in token_map:
                    d_inst.append(token_map[t])
                elif t.startswith('loc_'):
                    d_inst.append(token_map['LOC'])
                elif ':' in t:
                    d_inst.append(token_map['COMMA'])
                elif 'ptr' in t:
                    d_inst.append(token_map['PTR'])
                elif 'rva' in t:
                    d_inst.append(token_map['RVA'])
                elif 'offset sub_' in t:
                    d_inst.append(token_map['OFFSETSUB'])
                elif t.startswith('offset'):
                    d_inst.append(token_map['OFFSET'])
                elif t.startswith('sub_'):
                    d_inst.append(token_map['SUB'])
                elif t.startswith('word_'):
                    d_inst.append(token_map['WORD'])
                elif '[' in t:
                    d_inst.append(token_map['MEM'])
                else:
                    d_inst.append(token_map['VAL'])
            new_digit_block.append(d_inst)
        digit_basic_blocks.append(new_digit_block)

    print('begin to save')
    f = open(f_name, 'wb')
    cPickle.dump(digit_basic_blocks, f)
    cPickle.dump(token_map, f)
    f.close()
    f = open('token_map.pkl', 'wb')
    cPickle.dump(token_map, f)
    f.close()
    print('saved')
    return digit_basic_blocks, token_map