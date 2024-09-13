from features.prototype import *
import pefile


def extract_dll_feature(content):
    try:
        f = pefile.PE(data = content)
    except:
        return Counter()
    results = Counter()
    total = 0
    if not hasattr(f,'DIRECTORY_ENTRY_IMPORT'):
        return results
    for entry in f.DIRECTORY_ENTRY_IMPORT:
        results[entry.dll] = 1
        for imp in entry.imports:
            #results[entry.dll] += 1
            results[imp.name] = 1
            total += 1
    results['total'] = total
    return results
    

class import_dll_feature(feature_prototype):
    def __init__(self):
        super().__init__()
        #self.batch_normalization = True
        self.min_freq = 2000
        #print('self.min_freq:',self.min_freq)
    def scan(self,f_n, mini_len = 5):
        f = open(f_n,'rb')
        content = f.read()
        f.close()
        results = extract_dll_feature(content)
        for k in results.keys():
            self.counter[k] += 1
        if self.batch_normalization:
            self.update_feature_values(results)
        return results
    def scan_from_content(self,content, mini_len = 5):
        results = extract_dll_feature(content)
        for k in results.keys():
            self.counter[k] += 1
        if self.batch_normalization:
            self.update_feature_values(results)
        return results
    def scan_from_fea(self,results):
        for k in results.keys():
            self.counter[k] += 1
        if self.batch_normalization:
            self.update_feature_values(results)
        return results
        
        
    def convert(self, f):
        result = np.zeros((len(self.feature_map)+1))
        tmp = self.scan(f)
        rare = 0
        for k,v in tmp.items():
            if k in self.feature_map:
                result[self.feature_map[k]] = v
            else:
                rare += 1
        result[-1] = rare
        return list(result)
    
    def convert_from_fea(self,res):
        result = np.zeros((len(self.feature_map)+1))
        tmp = res
        rare = 0
        for k,v in tmp.items():
            if k in self.feature_map:
                result[self.feature_map[k]] = v
            else:
                rare += 1
        result[-1] = rare
        return list(result)