import random

from utils.utils import *
import traceback
import pickle as cPickle
import progressbar
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix
from models.clone_detection_model import *
from models.top_model import *
def get_samples(file_set_f):
    res =  set()
    files = os.listdir(file_set_f)
    for f in files:
        if not '.' in f:
            res.add(os.path.join(file_set_f,f))
    return res


def get_functions(fp,digi,ensure=False):
    if ensure and not ensure_file_type(fp):
        return False
    if ensure and not os.path.exists(fp+'.tmp0.json'):
        disassemble_one_file(fp)
    if not os.path.exists(fp+'.tmp0.json'):
        return False
    digit_functions = digi.digitalize_binary(fp+'.tmp0.json')
    return digit_functions

def put_f_in_buckets(fps,bin_func_reps):
    #buckets are just lists of file names
    buckets = {}
    for fp in fps:
        reps = bin_func_reps[fp]
        if type(reps) == int:
            leng = reps
        else:
            leng = len(reps)
        if leng not in buckets:
            buckets[leng] = []
        buckets[leng].append(fp)
    return buckets

def get_trained_feature_extractors(name):
    if not os.path.exists(name+'_feature_extractors'):
        return None
    feature_extractors = load_pickle(name+'_feature_extractors')
    return feature_extractors


def calc_func_reps_batch_mode(paras, fps, digi, name, model_name='starplus_transformer'):
    if not os.path.exists("./functmp/"):
        os.mkdir("./functmp/")
    with torch.no_grad():

        top_layer = 'cos'
        full_file_name = 'best_full_model' + model_name + '.pkl'
        block_level_name = 'block_level' + model_name + '.pkl'
        checkpoint = torch.load(full_file_name)
        if model_name == 'starplus_transformer':
            total_longest_block = torch.load(block_level_name).pos_emb.num_embeddings
        else:
            total_longest_block = torch.load(block_level_name).position_enc.num_embeddings
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        botmid_model = hierarchical_transformer(digi.token_map,n_src_vocab=len(digi.token_map), len_max_seq=total_longest_block,
                                                d_word_vec=paras.d_word_vec, \
                                                block_n_layers=paras.block_n_layers, function_n_layers=paras.function_n_layers,
                                                n_head=paras.n_head, d_k=paras.d_k, d_v=paras.d_v,
                                                d_model=paras.d_model, d_inner=paras.d_inner, n_hidden=paras.n_hidden, top_layer=top_layer,
                                                dropout=paras.dropout, model_name=model_name).to(device)

        botmid_model.load_state_dict(checkpoint['model_state_dict'])




        func_rep_path = 'func_rep_' + name + '/'
        name = 'functmp/' + name
        if not os.path.exists(func_rep_path):
            os.mkdir(func_rep_path)

        assert (type(fps) == list)
        bin_func_reps = {}

        n_fps = len(fps)

        split_n = 20
        split_points = [int(n_fps * i / split_n) for i in range(split_n)]
        print("split_points:",split_points)
        split_points.append(n_fps)
        for split_ind in range(len(split_points) - 1):
            print('split_ind:', split_ind)
            tmp_file_name = name + '_func_rep_tmp' + str(split_ind) + '.pkl'
            if os.path.exists(tmp_file_name):
                print('this split finishes')
                # bin_func_reps.update(load_pickle(tmp_file_name))
                continue

            fp_digi_func = {}
            block_buckets = {}
            func_buckets = {}
            func_n_block = {}
            block_cont = {}
            bin_n_func = {}

            if os.path.exists(name + '_digit_funcs' + str(split_ind) + '.pkl'):
                fp_digi_func = load_pickle(name + '_digit_funcs' + str(split_ind) + '.pkl')
            else:
                if os.path.exists(name + '_digit_funcstmp.pkl'):
                    fp_digi_func = load_pickle(name + '_digit_funcstmp' + str(split_ind) + '.pkl')
                bar = progressbar.ProgressBar(maxval=split_points[split_ind + 1] - split_points[split_ind],
                                              prefix='extract functions:').start()
                for i in range(split_points[split_ind], split_points[split_ind + 1]):
                    fp = fps[i]
                    print("fp:",fp)
                    if fp in fp_digi_func:
                        continue
                    bar.update(i - split_points[split_ind])
                    try:
                        funcs = get_functions(fp, digi)
                        print("funcs:",len(funcs))
                    except:
                        print('fp: {} no functions'.format(fp))
                        funcs = False
                    if funcs is False:
                        continue
                    fp_digi_func[fp] = funcs
                    if i % 500 == 0:
                        write_pickle(fp_digi_func, name + '_digit_funcstmp' + str(split_ind) + '.pkl')
                write_pickle(fp_digi_func, name + '_digit_funcs' + str(split_ind) + '.pkl')
            print('digit funcs got total:', len(fp_digi_func), 'functions', datetime.now().strftime("%H:%M:%S"))
            for fp, funcs in fp_digi_func.items():
                bin_n_func[fp] = len(funcs)
                for i, func in enumerate(funcs):
                    len_func = len(func)
                    func_n_block[fp + '_' + str(i)] = len_func
                    if len_func not in func_buckets:
                        func_buckets[len_func] = []
                    func_buckets[len_func].append(fp + '_' + str(i))
                    for j, block in enumerate(func):
                        block_cont[fp + '_' + str(i) + '_' + str(j)] = block
                        len_block = len(block)
                        if len_block not in block_buckets:
                            block_buckets[len_block] = []
                        block_buckets[len_block].append(fp + '_' + str(i) + '_' + str(j))

            block_rep = {}
            i = 0
            print('bucketed got total:', len(block_buckets), 'buckets', datetime.now().strftime("%H:%M:%S"))
            # for k,v in block_buckets.items():
            #    print('length of',k,'has',len(v),'functions')
            if os.path.exists(name + '_block_reps' + str(split_ind) + '.pkl'):
                block_rep = load_pickle(name + '_block_reps' + str(split_ind) + '.pkl')
            else:
                bar = progressbar.ProgressBar(maxval=len(block_buckets), prefix='block:').start()
                for blk_len, blocks in block_buckets.items():
                    t1 = time.time()
                    bar.update(i)
                    i += 1
                    n_sample = len(blocks)
                    present = 0
                    x = np.array([block_cont[block] for block in blocks])
                    if model_name == 'starplus_transformer':
                        batch_size = 5120
                    else:
                        if x.shape[1] <= 20:
                            batch_size = 10240
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        elif x.shape[1] > 20 and x.shape[1] <= 50:
                            batch_size = 2048
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        elif x.shape[1] > 50 and x.shape[1] <= 100:
                            batch_size = 512
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        elif x.shape[1] > 100 and x.shape[1] <= 200:
                            batch_size = 256
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        else:
                            batch_size = 64
                        print('length:', x.shape[1], 'batch_size:', batch_size)
                    while present < n_sample:
                        cur_inp = x[present:present + batch_size]
                        if model_name == 'starplus_transformer':
                            _, out = botmid_model.block_level(torch.tensor(cur_inp), device)
                            # print('np.array(cur_inp).shape:',np.array(cur_inp).shape)
                            # print('out.shape:',out.shape)
                        else:
                            out = botmid_model.block_level(torch.tensor(cur_inp), device)
                        out = out.cpu().detach().numpy()
                        for b, rep in zip(blocks[present:present + batch_size], out):
                            if model_name == 'starplus_transformer':
                                block_rep[b] = rep
                            else:
                                block_rep[b] = rep[0]
                        present += cur_inp.shape[0]
                    # print('length of {} with {} samples takes {} seconds'.format(blk_len,n_sample,time.time()-t1))
                write_pickle(block_rep, name + '_block_reps' + str(split_ind) + '.pkl')

            func_rep = {}
            i = 0

            print('len(bin_n_func):', len(bin_n_func))
            if os.path.exists(name + '_func_reps' + str(split_ind) + '.pkl'):
                func_rep = load_pickle(name + '_func_reps' + str(split_ind) + '.pkl')
            else:
                bar = progressbar.ProgressBar(maxval=len(func_buckets), prefix='func:').start()
                for func_len, funcs in func_buckets.items():
                    if func_len == 0:
                        continue
                    bar.update(i)
                    i += 1
                    n_sam = len(funcs)
                    present = 0
                    x = []
                    for func in funcs:
                        fun_blocks = []
                        for j in range(func_n_block[func]):
                            assert (func_n_block[func] == func_len)
                            fun_blocks.append(block_rep[func + '_' + str(j)])
                        x.append(fun_blocks)
                    x = np.array(x)
                    n_sample = len(x)
                    # print('actual func_len:',func_len, 'n samples:',len(x))
                    if model_name == 'starplus_transformer':
                        batch_size = 5120
                    else:
                        if x.shape[1] <= 20:
                            batch_size = 4096
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        elif x.shape[1] > 20 and x.shape[1] <= 50:
                            batch_size = 2048
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        elif x.shape[1] > 50 and x.shape[1] <= 100:
                            batch_size = 512
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        elif x.shape[1] > 100 and x.shape[1] <= 200:
                            batch_size = 256
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                        else:
                            batch_size = 64
                            print('length:', x.shape[1], 'batch_size:', batch_size)
                    while present < n_sample:
                        cur_inp = torch.tensor(x[present:present + batch_size])
                        mask = np.ones(cur_inp.shape[:2])
                        if model_name == 'starplus_transformer':
                            # print('cur_inp.shape:',cur_inp.shape)
                            _, out = botmid_model.function_level(cur_inp.to(device), torch.tensor(mask).to(device))
                        else:
                            out = botmid_model.function_level(cur_inp.to(device), torch.tensor(mask).to(device), digi.token_map['MASK'],
                                                              device)
                        out = out.cpu().detach().numpy()
                        for b, rep in zip(funcs[present:present + batch_size], out):
                            if model_name == 'starplus_transformer':
                                func_rep[b] = list(rep)
                            else:
                                func_rep[b] = list(rep[0])

                        present += cur_inp.shape[0]
                write_pickle(func_rep, name + '_func_reps' + str(split_ind) + '.pkl')
            tmp_bin_func_reps = {}
            total = 0
            for fp, n_fun in bin_n_func.items():
                if fp + '_' + str(n_fun - 1) not in func_rep:
                    # print(fp+'_'+str(n_fun-1),'not in func_rep')
                    total += 1
                else:
                    tmp_bin_func_reps[fp] = [func_rep[fp + '_' + str(i)] for i in range(n_fun)]
            print('total bins:', len(bin_n_func), 'n not have funcs:', total)
            write_pickle(tmp_bin_func_reps, tmp_file_name)

        bin_n_func = {}
        f = open('bin_n_func.txt', 'w')
        for split_ind in range(len(split_points) - 1):
            print('load split_ind:', split_ind)
            tmp_file_name = name + '_func_rep_tmp' + str(split_ind) + '.pkl'
            if os.path.exists(tmp_file_name):
                print("split_ind:",split_ind,"exist")
                print("tmp_file_name:",tmp_file_name)
                tmp_func_rep = load_pickle(tmp_file_name)
                print("len(tmp_func_rep):",len(tmp_func_rep))
                for k, v in tmp_func_rep.items():
                    f_save_add = os.path.join(func_rep_path, os.path.basename(k))
                    if not os.path.exists(f_save_add):
                        write_pickle(v, f_save_add)
                    n_fun = len(v)
                    bin_n_func[k] = n_fun
                    vec_len = len(v[0])
                    f.write(k + ' ' + str(n_fun) + '\n')
        f.close()

        zero_vector_rep = list(np.zeros((1, vec_len)))
        print('begin to add zero vector to fp with no function')
        f = open('bin_initialy_no_func.txt', 'w')
        for fp in fps:
            if fp not in bin_n_func:
                write_pickle(zero_vector_rep, os.path.join(func_rep_path, os.path.basename(fp)))
                bin_n_func[fp] = 1
                f.write(fp + '\n')
        f.close()
        return bin_n_func
def extractors_scan_files(feature_extractors,fps):
    nerror = 0
    for fp in fps:
        try:
            for fea in feature_extractors:
                fea.scan(fp)
        except Exception as e:
            nerror += 1
            print(fp,'failed to load')
            print(e)
            traceback.print_exc()
            continue
    print('number of fps:',len(fps),'number of errors:',nerror)
    print('scanned')
    for extractor in feature_extractors:
        extractor.prepare()
    print('prepared')


def one_train_step(top_model, epoch, funreps, features, func1_mask, y, optimizer, loss_func, individual_lossfunc,
                   cockpit, additional_loss='', additional_loss_weight=0):
    optimizer.zero_grad()
    # features = torch.tensor(features).to(device)
    # features is a list. further steps are done by the model
    # print("features.shape:",features.shape)
    if len(additional_loss) < 1:
        if not funreps is None:
            output = top_model(features, funreps, func1_mask)
        else:
            output = top_model(features)
    else:
        if not funreps is None:
            features.requires_grad_(True)
            funreps.requires_grad_(True)
            func1_mask.requires_grad_(True)
            inp, output, relevances = top_model(features, funreps, func1_mask, with_relevance=True)
        else:
            features.requires_grad_(True)
            inp, output, relevances = top_model(features, with_relevance=True)

    # output = top_model(features,funreps)
    if type(loss_func) == torch.nn.modules.loss.BCELoss:  # torch.nn.BCELoss
        y = torch.FloatTensor(np.array(y).astype(float)).cuda()  # .to(device)
    else:
        y = torch.LongTensor(y).cuda()  # .to(device)
    loss = loss_func(output, y)  # lb.transform(y)
    if len(additional_loss) >= 1:
        if len(output.shape) == 1 or output.shape[1] == 1:
            output = torch.stack([1 - output, output], dim=1)
        loss = loss + + additional_loss_weight * inben_robustness_loss(inp, output, inp.unsqueeze(-1), relevances)
    loss.backward()
    optimizer.step()
    return float(loss.data.detach().cpu().numpy().mean())


def one_test_step(replicas, x, device_ids):

    outputs = nn.parallel.parallel_apply(replicas, x, devices=device_ids)
    output = nn.parallel.gather(outputs, target_device=torch.device('cuda:0'))
    pred = np.round(output.detach().cpu().numpy())
    return pred


def train(top_model, buckets, bin_n_func, f_features, f_labels, optimizer, loss_func, individual_lossfunc, device, name,
          epoch, setname, cockpit=None, save_bucket_rep=False, use_func=True, additional_loss='',
          additional_loss_weight=0):
    top_model = top_model.cuda()
    top_model.train()
    top_model = nn.DataParallel(top_model).cuda()
    f = open(name + '_log.txt', 'w')
    f.close()

    if save_bucket_rep:
        funrep_buck_dir = './funrep_buck/'
        os.makedirs(funrep_buck_dir, exist_ok=True)

    use_gpu = True
    buck_ind = 0
    losses = []
    t1 = time.time()

    func_rep_path = 'func_rep_' + name + '/'
    n_bucks = len(buckets)
    print("n buckets:", n_bucks)
    for l, samples_ in buckets.items():
        samples, features_all = samples_
        print("bucket n func:", l, "n samples:", len(samples))
        # samples.sort()
        if save_bucket_rep:
            buck_file_name = funrep_buck_dir + name + '_' + setname + '_' + str(l) + '.pkl'
            if os.path.exists(buck_file_name):
                bin_func_rep, sam_names = load_pickle_list(buck_file_name, 2)
                for nam1, nam2 in zip(samples, sam_names):
                    if nam1 != nam2:
                        print('len(samples),len(sam_names):', len(samples), len(sam_names))
                        print('nam1,nam2:', nam1, nam2)
                    assert (nam1 == nam2)
            else:
                bin_func_rep = [load_pickle(os.path.join(func_rep_path, os.path.basename(sam))) for sam in samples]
                write_pickle_list([bin_func_rep, samples], buck_file_name)

        # bar.update(buck_ind)
        buck_ind += 1
        f = open(name + '_log.txt', 'a')
        f.write('epoch:{} bucket id: {} training samples of {} functions\n'.format(epoch, buck_ind, l))
        f.close()
        if hasattr(top_model, 'model_name') and top_model.model_name == 'transformer':
            if l <= 20:
                batch_size = 2048
            elif l > 20 and l <= 50:
                batch_size = 2048
            elif l > 200 and l <= 1000:
                batch_size = 1024
            elif l > 1000 and l <= 2000:
                batch_size = 32
            elif l > 2000 and l <= 2500:
                batch_size = 16
            elif l > 2500 and l <= 3000:
                batch_size = 8
        if hasattr(top_model, 'model_name') and top_model.model_name == 'malconv':
            batch_size = 36
        elif hasattr(top_model, 'model_name') and top_model.model_name == 'conviclr2018':
            batch_size = 128
        else:
            if l > 1000 and l < 2000:
                batch_size = 256
            elif l >= 2000 and l < 3000:
                batch_size = 128
            elif l >= 3000 and l < 7000:
                batch_size = 32
            elif l >= 7000:
                batch_size = 8
            else:
                batch_size = 5120
        present = 0
        n_sample = len(samples)

        bar = progressbar.ProgressBar(max_value=len(samples),
                                      prefix='epoch:' + str(epoch) + 'train bucket:' + str(buck_ind) + '/' + str(
                                          n_bucks) + ' of ' + str(l) + ' functions')
        # batch_size = 128
        while present < n_sample:
            # t1 = time.time()
            bar.update(present)
            if bin_n_func is not None:
                if save_bucket_rep:
                    funreps = bin_func_rep[present:present + batch_size]
                else:
                    funreps = []
                    for sam in samples[present:present + batch_size]:
                        try:
                            funrep = load_pickle(os.path.join(func_rep_path, os.path.basename(sam)))
                            funreps.append(funrep)
                        except:
                            print('{} got wrong funcrep'.format(os.path.join(func_rep_path, os.path.basename(sam))))
                            funreps.append(funreps[-1])
                            print(
                                '{} could replace it'.format(os.path.join(func_rep_path, os.path.basename(samples[0]))))
                funreps = torch.tensor(funreps, dtype=torch.float32).cuda()  # .to(device)
                func1_mask = torch.ones(funreps.shape[:2], dtype=torch.long).cuda()  # .to(device)

            else:
                funreps = None
                func1_mask = None
            features = torch.tensor(features_all[present:present + batch_size],
                                    dtype=torch.float32).cuda()  # .to(device)
            y = np.array([f_labels[sam] for sam in samples[present:present + batch_size]])
            if len(y) == 1:
                present += 1
                continue
            loss = one_train_step(top_model, epoch, funreps, features, func1_mask, y, optimizer, loss_func,
                                  individual_lossfunc, cockpit, additional_loss=additional_loss,
                                  additional_loss_weight=additional_loss_weight)
            losses.append(loss)
            present += len(features)
    avg_loss = torch.mean(torch.tensor(losses))
    print(' training loss: ' + str(avg_loss) + 'takes: ' + str(time.time() - t1))
    return avg_loss


def test(top_model, buckets, bin_n_func, f_features, f_labels, device, name, epoch, setname, save_bucket_rep=False,
         use_func=True, device_ids=[0, 1, 2, 3]):
    top_model.eval()

    res = {}
    trus = {}

    if save_bucket_rep:
        funrep_buck_dir = './funrep_buck/'
        os.makedirs(funrep_buck_dir, exist_ok=True)
    use_gpu = True
    full_samples = 0
    correct = 0
    t1 = time.time()
    f = open(name + '_log.txt', 'a')
    buck_ind = 0
    bar = progressbar.ProgressBar(max_value=len(buckets), prefix='epoch:' + str(epoch) + 'test ' + setname + ':')
    func_rep_path = 'func_rep_' + name + '/'
    # print("len(buckets):",len(buckets))

    full_time = 0
    full_sample = 0
    for l, samples_ in buckets.items():
        samples, features_all = samples_
        print("bucket n func:", l, "n samples:", len(samples))
        # samples.sort()
        f.write('epoch:{} training samples of {} functions\n'.format(epoch, l))
        bar.update(buck_ind)
        buck_ind += 1
        full_samples += len(samples)
        if save_bucket_rep:
            buck_file_name = funrep_buck_dir + name + '_' + setname + '_' + str(l) + '.pkl'
            if os.path.exists(buck_file_name):
                bin_func_rep, sam_names = load_pickle_list(buck_file_name, 2)
                for nam1, nam2 in zip(samples, sam_names):
                    assert (nam1 == nam2)
            else:
                bin_func_rep = [load_pickle(os.path.join(func_rep_path, os.path.basename(sam))) for sam in samples]
                # write_pickle_list([bin_func_rep,samples],buck_file_name)
        if l <= 20:
            batch_size = 2048
        elif l > 20 and l <= 50:
            batch_size = 1024
        elif l > 50 and l <= 200:
            batch_size = 64
        elif l > 200 and l <= 1000:
            batch_size = 32
        else:
            batch_size = 1

        if l > 9300:
            device = torch.device('cpu')
            if use_gpu:
                top_model = top_model.to(device)
            use_gpu = False
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if not use_gpu:
                top_model = top_model.to(device)
            use_gpu = True

        present = 0
        n_sample = len(samples)
        present = 0
        replicas = nn.parallel.replicate(top_model, device_ids)
        while present < n_sample:
            if bin_n_func is not None:
                if save_bucket_rep:
                    funreps = bin_func_rep[present:present + batch_size]
                else:
                    funreps = []
                    for sam in samples[present:present + batch_size]:
                        try:
                            funrep = load_pickle(os.path.join(func_rep_path, os.path.basename(sam)))
                            funreps.append(funrep)
                        except:
                            print('{} got wrong funcrep'.format(os.path.join(func_rep_path, os.path.basename(sam))))
                            funreps.append(funreps[-1])
                            print(
                                '{} could replace it'.format(os.path.join(func_rep_path, os.path.basename(samples[0]))))
            else:
                funreps = None
            features = torch.tensor(features_all[present:present + batch_size], dtype=torch.float32).to(device)
            if f_labels is not None:
                y = np.array([f_labels[sam] for sam in samples[present:present + batch_size]])
            present += len(features)
            features = nn.parallel.scatter(features, device_ids)
            if funreps is not None:
                funreps = torch.tensor(funreps, dtype=torch.float32).to(device)

                func1_mask = torch.ones(funreps.shape[:2], dtype=torch.long).to(device)

                funreps = nn.parallel.scatter(funreps, device_ids)
                func1_mask = nn.parallel.scatter(func1_mask, device_ids)
                x = [(fea, rep, mask) for fea, rep, mask in zip(features, funreps, func1_mask)]
            else:
                x = features

            models = replicas[:len(x)]
            devs = device_ids[:len(x)]

            t1 = time.time()

            pred = one_test_step(models, x, devs)
            t2 = time.time()
            n_sam = len(features)
            full_time += t2 - t1
            full_sample += n_sam
            # print("here 6")
            for sam, p, y0 in zip(samples[present:present + batch_size], pred, y):
                res[sam] = p
                trus[sam] = y0
            if f_labels is not None:
                correct += (pred == y).sum()
    if full_time == 0:
        print("full time is zero")
    else:
        print("n_samples per second:", full_sample / full_time)
    fps = []
    reals = []
    preds = []

    for k, v in res.items():
        fps.append(k)
        preds.append(v)
        reals.append(trus[k])

    print('epoch:' + str(epoch) + 'test ' + setname + ' accuracy:' + str(correct) + '/' + str(full_samples) + '=' + str(
        correct / full_samples) + 'takes: ' + str(time.time() - t1))
    f.close()
    print(confusion_matrix(reals, preds))
    return correct / full_samples, res, fps, reals, preds
def extractors_extract_features_from_files(feature_extractors, fps, name):
    f_no_features = []
    fea_path = 'fea_' + name + '/'
    print("here create features")

    if not os.path.exists(fea_path):
        os.mkdir(fea_path)
    f_features = feature_loader(fea_path)
    for fp in fps:
        # f_tmp = open(fp,'rb')
        # cont = f_tmp.read()
        # f_tmp.close()
        feature_in_list = []
        try:
            for i, fea in enumerate(feature_extractors):
                cur_feature = fea.convert(fp)
                feature_in_list.extend(cur_feature)
        except Exception as e:
            print(fp, 'failed to load')
            print(e)
            traceback.print_exc()
            f_no_features.append(fp + ' has wrong with feature' + str(type(feature_extractors[i])))
            continue
        write_pickle(feature_in_list, os.path.join(fea_path, os.path.basename(fp)))
    return f_features, f_no_features

def create_train_validation_test_for_detection(ben_base_path, mal_base_path,
                                               name, n_fold=5, time_split_train=None, time_split_test=None,
                                               time_split_train_ratio=None, target_root='E:/feature_folders/'):
    print('objs_for_' + name + '.pkl')
    print('objs_for_' + name + '_' + str(time_split_train) + '.pkl')
    if time_split_train is None and os.path.exists('objs_for_' + name + '.pkl'):
        fps, f_sets, f_label, name = load_pickle_list('objs_for_' + name + '.pkl', 4)
        return fps, f_sets, f_label, name
    elif time_split_train and os.path.exists('objs_for_' + name + '_' + str(time_split_train) + '_' + str(time_split_test) + '.pkl'):
        fps, f_set, f_label, name = load_pickle_list('objs_for_' + name + '_' + str(time_split_train) + '_' + str(time_split_test) + '.pkl', 4)
        return fps, f_set, f_label, name


    ben_file_set = get_samples(ben_base_path)
    mal_file_set = get_samples(mal_base_path)

    random.seed(0)

    ben_file_set = list(ben_file_set)
    random.shuffle(ben_file_set)
    mal_file_set = list(mal_file_set)
    random.shuffle(mal_file_set)

    fps = set()
    f_label = {}

    #fea_folder = 'fea_' + name + '/'
    #if not os.path.exists(fea_folder):
    #    os.mkdir(fea_folder)

    for f in ben_file_set:
        f_label[f] = 0
        fps.add(f)
    for f in mal_file_set:
        f_label[f] = 1
        fps.add(f)

    if time_split_train is None:
        f_sets = []
        n_ben_per_fold = int(len(ben_file_set)/n_fold)
        n_mal_per_fold = int(len(mal_file_set)/n_fold)
        ben_folds = []
        mal_folds = []

        for fold in range(n_fold):
            ben_folds.append(ben_file_set[n_ben_per_fold*fold:n_ben_per_fold*(1+fold)])
            mal_folds.append(mal_file_set[n_mal_per_fold*fold:n_mal_per_fold*(1+fold)])
        fold = 1
        for valid_fold in range(n_fold):
            for test_fold in range(n_fold):
                if valid_fold == test_fold:
                    continue
                valid_n = 0
                test_n = 0
                train_n = 0
                f_set = {}
                for f in ben_folds[valid_fold]:
                    f_set[f] = 'valid'
                    valid_n += 1
                for f in mal_folds[valid_fold]:
                    f_set[f] = 'valid'
                    valid_n += 1
                for f in ben_folds[test_fold]:
                    f_set[f] = 'test'
                    test_n += 1
                for f in mal_folds[test_fold]:
                    f_set[f] = 'test'
                    test_n += 1

                for train_fold in range(n_fold):
                    if train_fold == test_fold or train_fold == valid_fold:
                        continue
                    for f in ben_folds[train_fold]:
                        f_set[f] = 'train'
                        train_n += 1
                    for f in mal_folds[train_fold]:
                        f_set[f] = 'train'
                        train_n += 1
                f_sets.append(f_set)
                print('fold: {} train set: {} valid set: {} test set: {}'.format(fold,train_n, valid_n, test_n))
                fold += 1
        f_set_f_name = 'maldet_fset.pkl'
        write_pickle(f_sets, f_set_f_name)
        write_pickle_list([fps, f_sets, f_label, name], 'objs_for_' + name + '.pkl')
        return fps, f_sets, f_label, name

    else:
        timstamps = load_pickle('maldet_imad_timestamps.pkl')
        f_set = {}
        trainval_ben = []
        trainval_mal = []
        train_n = 0
        valid_n = 0
        test_n = 0
        for f in ben_file_set:
            if f in timstamps:
                f_year = datetime.fromtimestamp(timstamps[f]).year
                if f_year > 2020 or f_year < 2000:
                    continue
                if f_year > time_split_test:
                    f_set[f] = 'test'
                    test_n += 1
                elif f_year < time_split_train:
                    trainval_ben.append(f)

        n_ben = len(trainval_ben)
        for i, f in enumerate(trainval_ben):
            if i < int(time_split_train_ratio * n_ben):
                f_set[f] = 'train'
            else:
                f_set[f] = 'valid'

        for f in mal_file_set:
            if f in timstamps:
                f_year = datetime.fromtimestamp(timstamps[f]).year
                if f_year > 2020 or f_year < 2000:
                    continue
                if f_year > time_split_test:
                    f_set[f] = 'test'
                    test_n += 1
                elif f_year < time_split_train:
                    trainval_mal.append(f)

        n_mal = len(trainval_mal)
        for i, f in enumerate(trainval_mal):
            if i < int(time_split_train_ratio * n_mal):
                f_set[f] = 'train'
                train_n += 1
            else:
                f_set[f] = 'valid'
                valid_n += 1
        f_set_f_name = 'maldet_fset' + str(time_split_train) + '_' + str(time_split_test) + '.pkl'
        print('train set: {} valid set: {} test set: {}'.format(train_n, valid_n, test_n))
        write_pickle(f_set, f_set_f_name)
        write_pickle_list([fps, f_set, f_label, name], 'objs_for_' + name + '_' + str(time_split_train) + '_' + str(time_split_test) + '.pkl')

        return fps, f_set, f_label, name


class feature_loader(object):
    def __init__(self, folder):
        self.folder = folder

    def __getitem__(self, fp):
        fp = os.path.basename(fp)
        fp = os.path.join(self.folder, fp)
        res = load_pickle(fp)
        return res

    def get_fea_len(self):
        files = os.listdir(self.folder)
        print("self.folder:",self.folder)
        print("files:",files)
        fp = os.path.join(self.folder, files[0])
        res = load_pickle(fp)
        fea_len = len(res)
        return fea_len


def train_valid_test(pretrain, paras, fps, f_set, f_labels, name, daty, feature_extractors, deep_learning=True, use_func=True,
                     inter=True, devices=[0],
                     additional_loss='', additional_loss_weight=0):
    # consider whether add feature frequency as a parameter
    # final validation and test result write to name+'_result.txt'

    if not os.path.exists(name + '_label_to_ind.pkl'):
        print('compute label to ind')
        label_to_ind = {}
        label_set = set(f_labels.values())
        for i, lab in enumerate(label_set):
            label_to_ind[lab] = i

        write_pickle(label_to_ind, name + '_label_to_ind.pkl')

        # f = open(name+'_label_to_ind.pkl','wb')
        # cPickle.dump(label_to_ind,f)
        # f.close()
    else:
        print('load label to ind')
        label_to_ind = load_pickle(name + '_label_to_ind.pkl')
        # f = open(name+'_label_to_ind.pkl','rb')
        # label_to_ind = cPickle.load(f)
        # f.close()

    new_f_labels = {}
    for k, v in f_labels.items():
        ind = label_to_ind[v]
        new_f_labels[k] = ind
    f_labels = new_f_labels

    # feature_extractors = get_trained_feature_extractors(name)

    if not os.path.exists('fea_' + name + '/'):  # os.path.exists(name+'_f_features'):
        print('compute file features')
        if os.path.exists(name + '_feature_extractors'):
            print('load feature extractors')
            feature_extractors = get_trained_feature_extractors(name)
        else:
            print('create extractors')
            # feature_extractors = get_empty_feature_extractors()
            for fp in f_set:
                f_set[fp] == 'train'
            train_fps = [fp for fp in f_set if f_set[fp] == 'train']
            extractors_scan_files(feature_extractors, train_fps)

            f = open(name + '_feature_extractors', 'wb')
            cPickle.dump(feature_extractors, f)
            f.close()

        f_features, f_no_features = extractors_extract_features_from_files(feature_extractors, fps, name)
        f = open(name + 'f_no_features.txt', 'w')
        for line in f_no_features:
            f.write(line + '\n')
        f.close()

        print('number of files have no features:{}'.format(len(f_no_features)))
    else:
        print('load file features from ' + name + '_f_features')
        f_features = feature_loader('fea_' + name + '/')
    feature_length = f_features.get_fea_len()
    print('feature length:', feature_length)

    # fps = [fp for fp in fps if fp in f_features]
    fps = list(fps)
    # print('fps[:10]:',fps[:10])
    # print('list(f_set.keys())[:10]:',list(f_set.keys())[:10])

    train_fps = [fp for fp in f_set if f_set[fp] == 'train']
    valid_fps = [fp for fp in f_set if f_set[fp] == 'valid']
    test_fps = [fp for fp in f_set if f_set[fp] == 'test']
    print('len(f_set):', len(f_set))
    print('n_train:', len(train_fps))
    print('n_valid:', len(valid_fps))
    print('n_test:', len(test_fps))

    if use_func:
        digi = Digitalizer()
        if not os.path.exists(name + '_bin_n_func'):
            print('compute bin func rep')
            # bin_func_reps = calc_func_reps_non_batch_mode(fps,digi)

            print('calc_func_reps_batch_mode')
            bin_n_func = calc_func_reps_batch_mode(paras, fps, digi, name)
            write_pickle(bin_n_func, name + '_bin_n_func.pkl')
        else:
            print('load bin n func')
            bin_n_func = load_pickle(name + '_bin_n_func.pkl')
        print('number of binaries that have functions:', len(bin_n_func))
        print('number of binaries:', len(fps))
        print('len(train_fps):{},len(valid_fps):{},len(test_fps):{},total:{}'.format(len(train_fps), len(valid_fps),
                                                                                     len(test_fps),
                                                                                     len(train_fps) + len(
                                                                                         valid_fps) + len(test_fps)))
        print('len(bin_n_func):', len(bin_n_func))

        f = open('file_have_no_rep.txt', 'w')
        not_have_reps = 0
        for fp in fps:
            if fp not in bin_n_func:
                not_have_reps += 1
                f.write(fp + '\n')
        f.close()
        all_fps = set(fps)
        f = open('file_not_in_fps.txt', 'w')
        for fp in bin_n_func.keys():
            if fp not in all_fps:
                f.write(fp + '\n')
        f.close()
        print('number of fps not have reps:', not_have_reps)
    else:
        bin_n_func = None

    t1 = time.time()
    if use_func:
        print("here use func")
        train_buckets = put_f_in_buckets(train_fps, bin_n_func)
        if len(valid_fps) > 0:
            valid_buckets = put_f_in_buckets(valid_fps, bin_n_func)
        else:
            valid_buckets = {}
        test_buckets = put_f_in_buckets(test_fps, bin_n_func)
        print('put samples in buckets takes:{}'.format(time.time() - t1))
    else:
        print("here not use func")
        train_buckets = {0: train_fps}
        valid_buckets = {0: valid_fps}
        test_buckets = {0: test_fps}
    print("original n buckets:", len(train_buckets))
    new_train_buckets = {}
    new_valid_buckets = {}
    new_test_buckets = {}
    print("preparing features")
    if use_func:
        fea_buck = name + "_" + daty + "fea_bucks_use_func.pkl"
    else:
        fea_buck = name + "_" + daty + "fea_bucks" + ".pkl"
    t1 = time.time()
    if os.path.exists(fea_buck):
        train_buckets, valid_buckets, test_buckets = load_pickle_list(fea_buck, 3)
    else:
        old_bucks = [train_buckets, valid_buckets, test_buckets]
        new_bucks = [new_train_buckets, new_valid_buckets, new_test_buckets]
        for old_buck, new_buck in zip(old_bucks, new_bucks):
            for l, sams in old_buck.items():
                sams.sort()
                try:
                    features = [f_features[sam] for sam in
                                sams]  # np.array([f_features[sam] for sam in sams]).astype(np.float32)
                except:
                    lens = [len(f_features[sam]) for sam in sams]
                    print(lens)
                new_buck[l] = (sams, features)

        train_buckets = new_train_buckets
        valid_buckets = new_valid_buckets
        test_buckets = new_test_buckets
        write_pickle_list([train_buckets, valid_buckets, test_buckets], fea_buck)
    print("feature preparation takes:", time.time() - t1)
    print("new n buckets:", len(train_buckets))

    model_name = 'starplus_transformer'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    layer_dims = []
    top_model = IMAD_top(pretrain=pretrain, use_func=use_func, n_layers=paras.binary_n_layers, n_head=paras.n_head, d_k=paras.d_k, d_v=paras.d_v, \
                             d_model=3*paras.d_model, d_inner=paras.d_inner, other_feature_dim=feature_length, dropout=paras.dropout,
                             iffnn_hidden_dimensions=[int(i) for i in paras.iffnn_hidden_dims[1:-1].split(',')],
                             act_func=paras.act_func, use_batnorm=paras.use_batnorm,
                             use_last_norm=paras.use_last_norm, use_last_param=paras.use_last_param)
    binary_file_name = 'binary_model_'+model_name+'.pkl'
    if pretrain:
        if os.path.exists(binary_file_name):
            checkpoint = torch.load(binary_file_name)
            top_model.binary_level.load_state_dict(checkpoint['model_state_dict'])
            print("=====================\n Star-Galaxy Transformer loaded\n=====================")
        else:
            print("=====================\n Star-Galaxy Transformer not pretrained\n=====================")



    top_model = top_model.to(device)
    loss_func = torch.nn.BCELoss(reduction='mean')
    individual_lossfunc = torch.nn.BCELoss(reduction='none')

    model_parameters = filter(lambda p: p.requires_grad, top_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total number of parameters:{}'.format(params))
    print("feature_length:", feature_length)

    optimizer = torch.optim.Adam(top_model.parameters(), lr=8e-6)

    to_save_full_file_name = name + "_" + daty + '_topmodel.pkl'

    epochs = 1000
    best_acc = 0
    best_epoch = 0
    valid_acc = -1
    patience = 80

    reverse_map = {}
    for k, v in label_to_ind.items():
        print('k:{} v:{}'.format(k, v))
        if type(k) == str:
            reverse_map[v] = k.replace(' ', '_')
        else:
            reverse_map[v] = k
    t1 = time.time()
    total_epoch = 0

    cockpit = None
    losses = []
    train_accs = []
    best_train = 0
    test_acc = -1

    if not os.path.exists("./result/"):
        os.mkdir("./result/")

    for epoch in range(epochs):
        total_epoch += 1
        print('==========================================')
        print("epoch:", epoch)

        print("training:")

        avg_loss = train(top_model, train_buckets, bin_n_func, f_features, f_labels, optimizer, loss_func,
                         individual_lossfunc, device, name, epoch, 'train', cockpit,
                         additional_loss=additional_loss, additional_loss_weight=additional_loss_weight)
        losses.append(avg_loss)

        print("testing training set:")
        train_acc, res, fps, reals, preds = test(top_model, train_buckets, bin_n_func, f_features, f_labels, device,
                                                 name, epoch, 'train', use_func, device_ids=devices)
        if train_acc > best_train:
            best_train = train_acc
        train_accs.append(train_acc)
        print("testing valid set:")
        if len(valid_fps) > 0:
            valid_acc, res, fps, reals, preds = test(top_model, valid_buckets, bin_n_func, f_features, f_labels,
                                                     device, name, epoch, 'valid', use_func, device_ids=devices)

        print("testing test set:")
        acc, res, fps, reals, preds = test(top_model, test_buckets, bin_n_func, f_features, f_labels, device, name,
                                           epoch, 'test', use_func, device_ids=devices)
        if valid_acc > -1:
            print('valid_acc,best_acc:', valid_acc, best_acc)
            if valid_acc > best_acc:
                best_acc = valid_acc  # acc
                test_acc = acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': top_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, to_save_full_file_name)
                print('new best result')
                if pretrain:
                    torch.save({
                        'model_state_dict': top_model.binary_level.state_dict()
                    }, binary_file_name)

                preds = [reverse_map[it] for it in preds]
                reals = [reverse_map[it] for it in reals]
                write_to_result_file_list(test_fps, reals, preds, pre_fix=name)
        else:
            print('train_acc,best_acc:', train_acc, best_acc)
            if train_acc > best_acc:
                best_acc = train_acc
                best_epoch = epoch
                test_acc = acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': top_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, to_save_full_file_name)
                if pretrain:
                    torch.save({
                        'model_state_dict': top_model.binary_level.state_dict()
                    }, binary_file_name)
                print('new best result')
                preds = [reverse_map[it] for it in preds]
                reals = [reverse_map[it] for it in reals]
                write_to_result_file_list(test_fps, reals, preds, pre_fix=name)
        if epoch - best_epoch > patience:
            break
    total_time = time.time() - t1
    print("training time per epoch:", total_time / total_epoch)
    fig, ax1 = plt.subplots()
    ax1.plot(range(total_epoch), losses, 'g', label='Training loss')
    ax2 = ax1.twinx()
    ax2.plot(range(total_epoch), train_accs, 'b', label='Training acc')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print("best train:", best_train)
    print("best valid/train acc:", best_acc)
    print("best test acc:", test_acc)
    print("best_epoch:", best_epoch)
