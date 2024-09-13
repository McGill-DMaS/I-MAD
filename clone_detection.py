import os

import random
import pickle as cPickle


from models.clone_detection_model import *
from utils.utils import *

import argparse
import progressbar
import time


def get_func_clone_pair_dataset(negative_ratio=1, max_n_block=50, max_n_inst=50):
    f_name = 'clone_pair_dataset_unpadded_ratio' + str(negative_ratio) + '.pkl'
    if os.path.exists(f_name):
        f = open(f_name, 'rb')
        total_longest_block = cPickle.load(f)
        train = cPickle.load(f)
        validation = cPickle.load(f)
        test = cPickle.load(f)
        token_map = cPickle.load(f)
        f.close()
        print('loaded')
        return total_longest_block, train, validation, test, token_map

    function_map, full_func_groups = get_all_function_pair()
    train_ratio = 0.9
    valid_ratio = 0.05
    test_ratio = 0.05

    digi = Digitalizer()

    inputs = []
    masks = []
    outputs = []

    dataset = []

    for name, functions in full_func_groups:
        n_func = len(functions)
        for i in range(n_func):
            for j in range(i + 1, n_func):
                func1 = digi.digitalize_func(function_map[functions[i]])
                func2 = digi.digitalize_func(function_map[functions[j]])

                satisfy = True
                if len(func1) > max_n_block:
                    continue
                if len(func2) > max_n_block:
                    continue
                for block in func1:
                    if len(block) > max_n_inst:
                        satisfy = False
                        break
                if not satisfy:
                    continue
                for block in func2:
                    if len(block) > max_n_inst:
                        satisfy = False
                        break
                if not satisfy:
                    continue
                dataset.append((func1, func2, 1))

    n_pos = len(dataset)
    print('n_pos:', n_pos)

    func_ids = list(function_map.keys())
    n_func_id = len(func_ids)

    n_neg = 0
    while n_neg < int(n_pos * negative_ratio):
        fun1 = random.randint(0, n_func_id - 1)
        fun2 = random.randint(0, n_func_id - 1)
        func1 = digi.digitalize_func(function_map[func_ids[fun1]])
        func2 = digi.digitalize_func(function_map[func_ids[fun2]])
        satisfy = True
        if len(func1) > max_n_block:
            continue
        if len(func2) > max_n_block:
            continue
        for block in func1:
            if len(block) > max_n_inst:
                satisfy = False
                break
        if not satisfy:
            continue
        for block in func2:
            if len(block) > max_n_inst:
                satisfy = False
                break
        if not satisfy:
            continue
        dataset.append((func1, func2, -1))
        n_neg += 1
        if n_neg % 5000 == 0:
            print('currently n_neg', n_neg)

    print('n_neg:', n_neg)
    print('total:', len(dataset))

    random.shuffle(dataset)

    n_data = len(dataset)

    train = dataset[:int(train_ratio * n_data)]
    validation = dataset[int(train_ratio * n_data):int((train_ratio + valid_ratio) * n_data)]
    test = dataset[int((train_ratio + valid_ratio) * n_data):]

    total_longest_block = get_total_longest_block(function_map)

    token_map = digi.token_map

    to_dump = [total_longest_block, train, validation, test, token_map]
    f = open(f_name, 'wb')
    for it in to_dump:
        cPickle.dump(it, f)
    f.close()

    return total_longest_block, train, validation, test, token_map


def filter_with_size(dataset, n_block, n_inst):
    result = []
    for sample in dataset:
        func1 = sample[0]
        func2 = sample[1]
        satisfy = True
        if len(func1) > n_block:
            continue
        if len(func2) > n_block:
            continue
        for block in func1:
            if len(block) > n_inst:
                satisfy = False
                break
        if not satisfy:
            continue
        n_inst_total = 0
        for block in func1:
            n_inst_total += len(block)
        if n_inst_total <= 5:
            continue
        if not satisfy:
            continue
        for block in func2:
            if len(block) > n_inst:
                satisfy = False
                break
        if not satisfy:
            continue
        n_inst_total = 0
        for block in func2:
            n_inst_total += len(block)
        if n_inst_total <= 5:
            continue
        result.append(sample)
    return result


def test(model, data_set, batch_size, device):
    n_batch = int(len(data_set) / batch_size)
    pred = []
    model.eval()
    bar = progressbar.ProgressBar(maxval=len(range(n_batch)), prefix='test').start()
    for i in range(n_batch):
        bar.update(i + 1)
        func1 = []
        func2 = []
        y = []
        for sample in data_set[i * batch_size:(i + 1) * batch_size]:
            func1.append(sample[0])
            func2.append(sample[1])
            y.append(sample[2])
        out = model.forward(func1, func2, device)
        out = out.data.cpu().numpy()
        out = out.reshape(out.shape[0], )

        # fixed
        res = out > 0
        out = []
        for r in res:
            if r == True:
                out.append(1)
            else:
                out.append(-1)

        # out = list(np.round(out).astype(int))
        pred.extend(out)  # @UndefinedVariable
    if len(data_set) % batch_size > 0:
        func1 = []
        func2 = []
        y = []
        for sample in data_set[n_batch * batch_size:]:
            func1.append(sample[0])
            func2.append(sample[1])
            y.append(sample[2])
        out = model.forward(func1, func2, device)
        out = out.data.cpu().numpy()
        out = out.reshape(out.shape[0], )
        # fixed
        res = out > 0
        out = []
        for r in res:
            if r == True:
                out.append(1)
            else:
                out.append(-1)

        # out = list(np.round(out).astype(int))   this is the bug
        pred.extend(out)  # @UndefinedVariable

    pred = np.array(pred)
    real = np.array([sample[2] for sample in data_set])
    f = open('clone_detection_test.txt', 'w')
    for p, r in zip(pred, real):
        f.write(str(p) + ' ' + str(r) + '\n')
    f.close()
    return (pred == real).sum() / real.shape[0]


def functionlength(sample):
    length = max(len(sample[0]), len(sample[1]))
    return length



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='', required=True)
    parser.add_argument('--max_n_block', type=int, help='', required=True)
    parser.add_argument('--max_n_inst', type=int, help='', required=True)
    parser.add_argument('--d_word_vec', type=int, help='Dimension of token embeddings.', required=True)
    parser.add_argument('--block_n_layers', type=int, help='Number of layers at block level', required=True)
    parser.add_argument('--function_n_layers', type=int, help='Number of layers at function level', required=True)
    parser.add_argument('--n_head', type=int, help='Number of attention heads', required=True)
    parser.add_argument('--d_model', type=int, help='', required=True)
    parser.add_argument('--d_k', type=int, help='Dimension of key vectors', required=True)
    parser.add_argument('--d_v', type=int, help='Dimension of value vectors', required=True)
    parser.add_argument('--dropout', type=float, help='Dropout rate', required=True)
    parser.add_argument('--patience', type=int, help='Patience for early stopping', required=True)
    parser.add_argument('--d_inner', type=int, help='Dimension of hidden layer of positionwise feedforward neural '
                                                    'network', required=True)
    parser.add_argument('--n_hidden', type=int, help='Dimension of hidden layer to predict', required=True)
    parser.add_argument('--max_epochs', type=int, help='', required=True)


    paras = parser.parse_args()



    model_name = 'starplus_transformer'
    block_level_name = 'block_level' + model_name + '.pkl'
    vis = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    log_file = open('clone_log.txt', 'w')
    batch_size = paras.batch_size
    top_layer = 'cos'
    t1 = time.time()
    print('before get_dataset')
    total_longest_block, train_set, validation_set, test_set, token_map = get_func_clone_pair_dataset(1)
    print('load data takes:', time.time() - t1)
    print('token map size:', len(token_map))
    max_n_block = paras.max_n_block
    max_n_inst = paras.max_n_inst
    t1 = time.time()

    train_set = filter_with_size(train_set, max_n_block, max_n_inst)
    validation_set = filter_with_size(validation_set, max_n_block, max_n_inst)
    test_set = filter_with_size(test_set, max_n_block, max_n_inst)
    train_set.sort(key=functionlength)
    validation_set.sort(key=functionlength)
    test_set.sort(key=functionlength)

    print('filter data takes:', time.time() - t1)

    print('dataset size: train: {} validation: {} test: {}'.format(len(train_set), len(validation_set), len(test_set)))

    patience = paras.patience

    full_file_name = 'best_full_model' + model_name + '.pkl'

    verbose = True

    max_epochs = paras.max_epochs
    losses = []
    testAccs = []
    trainAccs = []

    if model_name == 'starplus_transformer':
        total_longest_block = torch.load(block_level_name).pos_emb.num_embeddings
    else:
        total_longest_block = torch.load(block_level_name).position_enc.num_embeddings

    print('total_longest_block:', total_longest_block)
    model = hierarchical_transformer(token_map, n_src_vocab=len(token_map), len_max_seq=total_longest_block, d_word_vec=paras.d_word_vec, \
                                     block_n_layers=paras.n_layers, function_n_layers=paras.n_layers, n_head=paras.n_head, d_k=paras.d_k, d_v=paras.d_v, d_model=paras.d_model \
                                     , d_inner=paras.d_inner, n_hidden=paras.n_hidden, top_layer=top_layer, dropout=paras.dropout, model_name=model_name).to(
        device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total number of parameters:{}'.format(params))
    print(model)

    print()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # @UndefinedVariable

    if top_layer == 'cos':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.BCELoss()

    init_epoch = 0

    load_block_level = False#True
    if load_block_level:
        model.block_level = torch.load(block_level_name).to(device)
        print('load block level success')

    load_full = False


    print('token_map:', len(token_map))
    if load_full and os.path.exists(full_file_name):
        checkpoint = torch.load(full_file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model = model.to(device)
        print('load full success')
    testAcc = test(model, test_set, batch_size, device)
    print('\ntest accuracy:', testAcc)

    best_epoch = 0
    best_valid_acc = 0


    for epoch in range(init_epoch, max_epochs):
        model.train()
        iteration = 0
        costVector = []
        # randomly choose n_batches index from 0 to n_batches-1 without repetition which is randomize the order of the list
        n_batch = int(len(train_set) / batch_size)
        t1 = time.time()
        bar = progressbar.ProgressBar(maxval=len(range(n_batch)), prefix='Epoch: ' + str(epoch) + ' train').start()
        for i in range(n_batch + 1):
            bar.update(i)
            func1 = []
            func2 = []
            y = []
            for sample in train_set[i * batch_size:(i + 1) * batch_size]:
                func1.append(sample[0])
                func2.append(sample[1])
                y.append([sample[2]])
            if top_layer == 'cos':
                y = Variable(torch.FloatTensor(y)).to(device).resize(len(y))
            else:
                y = Variable(torch.FloatTensor(y)).to(device)

            if len(func1) == 0:
                continue
            optimizer.zero_grad()


            out = model.forward(func1, func2, device)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            costVector.append(loss.item())
            if (iteration % 1 == 0) and verbose:
                losses.append(loss.item())
                # print('    Epoch:%d, Iteration:%d, Train_Cost:%f' % (epoch, iteration, loss.item()))
            iteration += 1
        trainCost = np.mean(costVector)

        model.eval()
        trainAcc = test(model, train_set, batch_size, device)
        validationAcc = test(model, validation_set, batch_size, device)
        testAcc = test(model, test_set, batch_size, device)

        if validationAcc > best_valid_acc:
            best_valid_acc = validationAcc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, full_file_name)
            best_epoch = epoch
            final_test = testAcc
        else:
            if epoch - best_epoch > patience:
                break

        buf = '\nEpoch:%d, time consuming: %f, Train_cost:%f Train accuracy:%f Validation accuracy:%f Test accuracy:%f' % (
        epoch, time.time() - t1, trainCost, trainAcc, validationAcc, testAcc)
        testAccs.append(testAcc)
        print(buf)
    print('best validation acc: {}, final test_set: {}'.format(best_valid_acc, final_test))