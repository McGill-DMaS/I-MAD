import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
import bz2

def write_to_result_file_list(fnames,reals,preds,pre_fix=''):
    f = open('result/'+pre_fix+'result_list.txt','w')
    for fn,real,pred in zip(fnames,reals,preds):
        nopath_fn = fn.split('\\')[-1]
        f.write(nopath_fn+" "+str(real)+" "+str(pred)+"\n")
    f.close()


def get_recursive_file_list(path):
    current_files = os.listdir(path)
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        if os.path.isdir(full_file_name):
            next_level_files = get_recursive_file_list(full_file_name)
            all_files.extend(next_level_files)
        else:
            all_files.append(full_file_name)
    return all_files

def load_pickle(fn,compressed=True):
    if not compressed:
        f = open(fn,'rb')
    else:
        f = bz2.BZ2File(fn, 'rb')
    try:
        res = pickle.load(f)
    except:
        f.close()
        if compressed:
            f = bz2.BZ2File(fn, 'rb')
            res = pickle.load(f)
            f.close()
        else:
            f = open(fn,'rb')
            res = pickle.load(f)
            f.close()
            write_pickle(res,fn,True)
        return res
    f.close()
    return res


def get_total_longest_block(function_map):
    total_longest_block = 0
    for k,v in function_map.items():
        blocks = v['blocks']
        for block in blocks:
            leng = len(block['src']) + 1
            if leng > total_longest_block:
                total_longest_block = leng
    return total_longest_block

class Digitalizer(object):
    def __init__(self, fn=None):
        if fn is None:
            fn = 'token_map.pkl'
        # f = open(fn,'rb')
        # self.token_map = cPickle.load(f)
        # f.close()
        self.token_map = load_token_map()

    def digitalize_block(self, block, black_length_min=5, black_length_max=1000):
        # block should be a list of src already each inst starts with add
        if len(block) < 5 or len(block) > 1000:
            return None
        new_digit_block = []
        token_map = self.token_map
        for inst in block:
            d_inst = []
            for t in inst[1:4]:
                if t in self.token_map:
                    d_inst.append(self.token_map[t])
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

            d_inst = d_inst + [self.token_map['EMPTY_OPRA']] * (3 - len(d_inst))
            new_digit_block.append(d_inst)
        return new_digit_block

    def digitalize_func(self, func, func_length_min=1, black_length_min=5, black_length_max=1000):
        # func should be a dict from the json file
        result = []
        blocks = func['blocks']
        if len(blocks) < func_length_min:
            return None
        for block in blocks:
            src = block['src']
            di_bl = self.digitalize_block(src, black_length_min, black_length_max)
            if di_bl is not None:
                result.append(di_bl)
        return result

    def digitalize_binary(self, fn, func_length_min=1, black_length_min=5, black_length_max=1000):
        # fn should be the path to a json file
        con = open(fn, 'r', encoding='utf-8')
        data = json.loads(con.read())
        functions = data['functions']
        digit_functions = []
        for func in functions:
            res = self.digitalize_func(func, func_length_min, black_length_min, black_length_max)
            if len(res) > 0:
                digit_functions.append(res)
        return digit_functions

def save_token_map(token_map):
    write_pickle(token_map,'token_map.pkl')
def load_token_map():
    token_map = load_pickle('token_map.pkl')
    return token_map

def function_clone_pair(functions):
    func_dic = {}
    for func in functions:
        if func['name'] in func_dic:
            func_dic[func['name']].append(func['id'])
        else:
            func_dic[func['name']] = [func['id']]
    return func_dic

def filter_function(function, n_inst=5):
    r_n = 0
    blocks = function['blocks']
    for block in blocks:
        r_n += len(block['src'])
    return r_n > n_inst

def read_assembly_functions_from_json(f):
    json_data=open(f,encoding='utf-8').read()
    data = json.loads(json_data)
    functions=data['functions']
    result = []
    for func in functions:
        if filter_function(func):
            result.append(func)
    return result


def get_all_function_pair():
    '''
    first return id->original func
    second return name->[id1,id2,...]
    '''
    inst_set = ()
    paths = ['./data/clone_detection/obf/','./data/clone_detection/opt/']
    function_map = {}
    full_func_groups = []
    for path in paths:
        current_folders = os.listdir(path)
        for folder in current_folders:
            func_group = {}
            files = os.listdir(path+folder+'/')
            tmp_functions = []
            for file in files:
                functions = read_assembly_functions_from_json(path+folder+'/'+file)
                tmp_functions.extend(functions)
                for func in functions:
                    assert func['id'] not in function_map
                    function_map[func['id']] = func
            func_dic = function_clone_pair(tmp_functions)
            full_func_groups.extend(func_dic.items())
    return function_map, full_func_groups


def write_pickle(obj,fn,compressed=True):
    if not compressed:
        f = open(fn,'wb')
    else:    
        f = bz2.BZ2File(fn, 'wb')
    pickle.dump(obj,f)
    f.close()
    return
def load_pickle_list(fn,n,compressed=True):
    if not compressed:
        f = open(fn,'rb')
    else:
        f = bz2.BZ2File(fn, 'rb')
    try:
        res = []
        for i in range(n):
            tmp = pickle.load(f)
            res.append(tmp)
    except:
        if compressed:
            f = bz2.BZ2File(fn, 'rb')
        else:
            f = open(fn,'rb')
        res = []
        for i in range(n):
            tmp = pickle.load(f)
            res.append(tmp)
        f.close()
        if compressed:
            write_pickle(res,fn,True)
        return res
            
    f.close()
    return res

def write_pickle_list(objs,fn,compressed=True):
    if not compressed:
        f = open(fn,'wb')
    else:
        f = bz2.BZ2File(fn, 'wb')
    for obj in objs:
        pickle.dump(obj,f)
    f.close()
    return    

def write_confusion_matrix(f,conf):
    for row in conf:
        sum = row.sum()
        largest = row.argmax()
        for i,ele in enumerate(row):
            if i == largest:
                f.write('& \\textbf{{{}({:.1f}\\%)}}'.format(ele,ele/sum*100))
            else:
                f.write('& {}({:.1f}\\%)'.format(ele,ele/sum*100))
        f.write('\\\\\n')



