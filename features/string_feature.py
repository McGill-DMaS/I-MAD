from features.prototype import *
import re
import pefile



def parse_strings(content,mini_len):
    strings = []
    state = 0 #0 out 1 in 2 final
    total = 0
    counter = Counter()
    for i in range(len(content)):
        if content[i] > 0 and content[i] <127:
            if state == 0:
                beg = i
                state = 1
        elif content[i] == 0:
            if state == 1:
                if i-beg<mini_len:
                    state = 0
                    continue
                cur = content[beg:i].decode()
                counter[cur] += 1
                total += 1
                strings.append(cur)
                state = 0
            else:
                state = 0
        else:
            state = 0
    result = Counter(strings)
    return result, total, counter

ASCII_BYTE = b" !\"#\$%&\'\(\)\*\+,-\./0123456789:;<=>\?@ABCDEFGHIJKLMNOPQRSTUVWXYZ\[\]\^_`abcdefghijklmnopqrstuvwxyz\{\|\}\\\~\t"
re_narrow = re.compile(b'([%s]{%d,})' % (ASCII_BYTE, 8))
re_wide = re.compile(b'((?:[%s]\x00){%d,})' % (ASCII_BYTE, 8))

# Process a given PE file
def parse_strings2(file_bytes):
    strings = []
    ignore_strings = []
    counter = Counter()
    try:
        
        pe = pefile.PE(data = file_bytes)
        
        # Get strings we're not interested in
        #with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ignore_strings.txt")) as f:
        #    ignore_strings = f.read().splitlines()
        #
        #    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ignore_apis.txt")) as f:
        #        ignore_apis = f.read().splitlines()
        #        ignore_strings = ignore_strings + ignore_apis
        
        # Get import strings to ignore
        #if 'DIRECTORY_ENTRY_IMPORT' in pe.__dict__:
        #    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        #        for func in entry.imports:
        #            if func and func.name:
        #                ignore_strings.append(func.name.decode('utf-8'))

        # Go through each section pulling out strings
        for section in pe.sections:  
            is_dotnet = False
            if pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR']].VirtualAddress != 0:
                    is_dotnet = True

             # Ignore executable sections if the file isn't .Net (strings are in .text for .Net)
            if is_dotnet or (not section.__dict__.get('IMAGE_SCN_MEM_EXECUTE', False)):
                # Ignore certain secions
                #TODO: Extract resources seperately..?
                if not any(ig_sect in section.Name.decode('UTF-8') for ig_sect in (".reloc", ".rsrc")): 
                    tmp_strings = []

                    data_start = (section.PointerToRawData)
                    data_len = (section.SizeOfRawData)
                    b = file_bytes[data_start:data_start+data_len]

                    for match in re_narrow.finditer(b):
                        tmp_strings.append(match.group().decode('ascii').strip())
                    for match in re_wide.finditer(b):
                        try:
                            tmp_strings.append(match.group().decode('utf-16').strip())
                        except UnicodeDecodeError:
                            pass
                    
                    for s in tmp_strings:
                        if len(s) > 0 and len(s) < 200 and s not in ignore_strings:
                            counter[s.lower()] += 1
                            total += 1
                            strings.append(s.lower())
    except:
        pass

    result = Counter(strings)
    return result, total, counter

def extract_string_features(content, mini_len = 5):
    result, total, counter = parse_strings(content, mini_len = 5)
    result['total'] = total
    return result

class string_feature(feature_prototype):
    def scan(self,f_n, mini_len = 5):
        f = open(f_n,'rb')
        content = f.read()
        f.close()
        results = extract_string_features(content)
        for k in results.keys():
            self.counter[k] += 1
        if self.batch_normalization:
            self.update_feature_values(results)
        return results
    def scan_from_content(self,content, mini_len = 5):
        result = extract_string_features(content)
        for k in result.keys():
            self.counter[k] += 1
        if self.batch_normalization:
            self.update_feature_values(result)
        return result
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
    def convert_from_fea(self, res):
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
    
    def scan_from_fea(self,results):
        for k in results.keys():
            self.counter[k] += 1
        if self.batch_normalization:
            self.update_feature_values(results)
        return results
