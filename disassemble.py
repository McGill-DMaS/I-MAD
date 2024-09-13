import magic
import os
import gzip
import progressbar
from subprocess import call
import sys
from multiprocessing import Pool


def get_recursive_file_list(path): 
    current_files = os.listdir(path)  
    all_files = []  
    for file_name in current_files:  
        full_file_name = os.path.join(path, file_name)  
        all_files.append(full_file_name)  
        if os.path.isdir(full_file_name):  
            next_level_files = get_recursive_file_list(full_file_name)  
            all_files.extend(next_level_files)  
    return all_files  
def disassemble_one_file(fp):
    IDA_script = os.path.join(os.getcwd()+"\\ExtractBinaryViaIDA",'ExtractBinaryViaIDA.py')
    call(
        ['ida64', '-A', '-S{}'.format(IDA_script), fp])
def disassemble_all_files(base_path):
    all_files = get_recursive_file_list(base_path)
    total = len(all_files)
    print('total:',total)
    bar = progressbar.ProgressBar(total)
    p = Pool(10) 
    i = 0
    for fp in all_files:
        i += 1
        bar.update(i)
        if fp.endswith('.json') or fp.endswith('.i64'):
            continue
        if os.path.exists(fp+'.tmp0.json'):
            continue
        p.apply_async(disassemble_one_file, args=(fp,))


if __name__ == '__main__':
    data_root = './data/'
    disassemble_all_files(data_root)