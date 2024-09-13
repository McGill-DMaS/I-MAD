from collections import Counter
import numpy as np
class feature_prototype(object):

    def __init__(self, batch_normalization = False, min_freq = 1000):
        self.counter = Counter()
        self.feature_map = {}
        self.min_freq = min_freq
        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.feature_values = {}
    def scan(self,f):
        raise NotImplementedError
    def prepare(self):
        i = 0
        if self.batch_normalization:
            self.fea_avg = {}
            self.var = {}
        #print('prepare counter size:',len(self.counter))
        for k,v in self.counter.items():
            if v >= self.min_freq:
                self.feature_map[k] = i
                i += 1
                if self.batch_normalization:
                    l = np.array(self.feature_values[k])
                    self.fea_avg[k] = np.mean(l)
                    self.var[k] = np.var(l)
        #print('extractor prepared, minimal frequency: {},i:{}, feature map size: {}'.format(self.min_freq,i,self.feature_map))
        return
    def update_feature_values(self,res):
        for k,v in res.items():
            if not k in self.feature_values:
                self.feature_values[k] = []
            self.feature_values[k].append(v)
    def convert(self, f):
        result = np.zeros((len(self.feature_map)))
        tmp = self.scan(f)
        for k,v in tmp.items():
            if k in self.feature_map:
                if self.batch_normalization:
                    result[self.feature_map[k]] = (v-self.fea_avg[k])/np.sqrt(self.var[k]**2+1e-6)
                else:
                    result[self.feature_map[k]] = v
                    
        return list(result)
    def convert_from_fea(self,res):
        result = np.zeros((len(self.feature_map)))
        tmp = res
        for k,v in tmp.items():
            if k in self.feature_map:
                if self.batch_normalization:
                    result[self.feature_map[k]] = (v-self.fea_avg[k])/np.sqrt(self.var[k]**2+1e-6)
                else:
                    result[self.feature_map[k]] = v
                    
        return list(result)