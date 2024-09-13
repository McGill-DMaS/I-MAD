import os
import progressbar
import pickle
import gzip
os.sys.path.append('./features/')
from header_real_value_feature import extract_infos,header_real_value_feature
from import_dll_feature import extract_dll_feature,import_dll_feature
from string_feature import extract_string_features,string_feature
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import *


def extract_a_file(source_folders, fp,features_to_extract, features_name_to_extract,target_root):
    res = []
    try:
        for path,typ in source_folders: 
            full_path = os.path.join(path, fp)  
            print('full_path:',full_path)
            if not os.path.exists(full_path):
                continue
            if typ == 'zipped':
                f = gzip.open(full_path,'rb')
                content = f.read()
                f.close()
            else:
                assert(typ == 'unzipped')
                f = open(full_path,'rb')
                content = f.read()
                f.close()
            for i,(ex,exn) in enumerate(zip(features_to_extract, features_name_to_extract)):
                fea = ex(content)
                write_pickle(fea,target_root+exn+'/'+fp,True)
            return
    except Exception as e:
        print('str(e):\t\t', str(e))
        return
    return

def extract_features_from_specification(file_names,source_folders,features_to_extract,features_name_to_extract,result_name,target_root = 'E:/feature_folders/'):
    
    
    for exn in features_name_to_extract:
        path = target_root+exn
        if not os.path.exists(path):
            os.mkdir(path)
    
    #bar = progressbar.ProgressBar(len(file_names))
    print('number of files to scan: {}'.format(len(file_names)))
    n_jobs=20
    with Parallel(n_jobs=n_jobs, verbose=1) as parallel:
        result_list = parallel(
            delayed(extract_a_file)(source_folders, fp,features_to_extract,features_name_to_extract,target_root) for
            fp in file_names)
def write_feature_vecotr(f,f_root,fea_path,ext_n):
    
    fea0 = load_pickle(f_root+f,True)
    extractor = load_pickle(ext_n)
    fea_ = extractor.convert_from_fea(fea0)
    write_pickle(fea_,fea_path+f,True)
    return
    
def prepare_feature_extractors_with_extracted_features(file_names,feature_extractors,features_name_to_extract,result_names,target_root = 'E:/feature_folders/',pass_scan = False):
    for extractor, fea in zip(feature_extractors,features_name_to_extract):
        
        ext_name = str(type(extractor)).split('\'')[1].split('.')[1]
        fea_path = target_root+ext_name+'/'
        if not os.path.exists(fea_path):
            os.mkdir(fea_path)
        
        f_root = target_root+fea+'/'
        files = os.listdir(f_root)
        print('feature: {} total files:{}'.format(fea,len(files)))
        n_scanned = 0
        if not pass_scan:
            for f in tqdm(files):
                if f in file_names:
                    n_scanned += 1
                    fea_ = load_pickle(f_root+f,True)
                    extractor.scan_from_fea(fea_)
            print('n scanned:{}'.format(n_scanned))
            print('feature {} scanned'.format(fea))
        extractor.prepare()
        ext_n = '_'.join(result_names)+'___'+str(type(extractor)).split('\'')[1].split('.')[1]+'.pkl'
        write_pickle(extractor,ext_n)

        for f in tqdm(files):
            if f in file_names:
                fea0 = load_pickle(f_root+f,True)
                fea_ = extractor.convert_from_fea(fea0)
                write_pickle(fea_,fea_path+f,True)

if __name__ == '__main__':
    path = './data/malicious_binaries/'
    base_name = os.path.basename(os.path.normpath(path))
    result_name1 = base_name
    picked_files = load_pickle(base_name+'_picked_files.pkl')
    
    path = './data/all_benign_software/'
    base_name = os.path.basename(os.path.normpath(path))
    result_name2 = base_name
    picked_files = picked_files.union(load_pickle(base_name+'_picked_files.pkl'))
    
    result_name = result_name1+'_'+result_name2
    
    
    source_folders = [('./data/malicious_binaries/','unzipped'),('./data/all_benign_software/','unzipped')]
    features_to_extract = [extract_infos,extract_dll_feature,extract_string_features]
    features_name_to_extract = ['extract_infos', 'extract_dll_feature', 'extract_string_features']
    extract_features_from_specification(picked_files,source_folders,features_to_extract,features_name_to_extract,result_name)           