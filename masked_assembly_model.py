import os
os.sys.path.append('../../')
from torch.utils.data import Dataset

import time

from sklearn import preprocessing

from models.basic_models import *
from utils.masked_assembly_model_utils import *

import argparse
import progressbar

#CHECKED
def load_masked_am_dataset(paths):
    if os.path.exists('masked_am_dataset.pkl'):
        t1 = time.time()
        f = open('masked_am_dataset.pkl', 'rb')
        inputs = cPickle.load(f)
        masks = cPickle.load(f)
        outputs = cPickle.load(f)
        token_map = cPickle.load(f)
        f.close()
        print('load masked am dataset takes:', time.time() - t1)
        return inputs, masks, outputs, token_map

    t1 = time.time()
    blocks, token_map = get_digit_basic_block_dataset(maximum_length=50, tokenmap_size=1500, paths=paths)
    print('digitalization takes:', time.time() - t1)
    print('token map size:', len(token_map))

    t1 = time.time()
    min_n_inst = 5
    max_n_inst = 250
    filtered = []
    print('before filter ', len(blocks), ' blocks')
    for block in blocks:
        if len(block) >= min_n_inst and len(block) <= max_n_inst:
            filtered.append(block)
    blocks = filtered
    print('after filter ', len(blocks), ' blocks')
    print('sorting takes:', time.time() - t1)

    t1 = time.time()
    blocks.sort(key=len)
    print('sort takes:', time.time() - t1)

    inputs, masks, outputs = get_mask_data_from_digit_blocks(blocks, token_map)
    # oov is not trained
    print('token map size:', len(token_map))

    f = open('masked_am_dataset.pkl', 'wb')
    cPickle.dump(inputs, f)
    cPickle.dump(masks, f)
    cPickle.dump(outputs, f)
    cPickle.dump(token_map, f)
    f.close()
    f = open('token_map.pkl', 'wb')
    cPickle.dump(token_map, f)
    f.close()
    return inputs, masks, outputs, token_map


class MASKDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, inputs, masks, outputs):
        'Initialization'
        self.inputs = inputs
        self.masks = masks
        self.outputs = outputs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.inputs[index]
        m = self.masks[index]
        y = self.outputs[index]

        return x, y, m

def train_one_step(x, m, device, model, criterion, optimizer):
    optimizer.zero_grad()
    output = model(x, m, device)
    loss = criterion(output, Variable(torch.LongTensor(y)).to(device))  # lb.transform(y)
    loss.backward()
    optimizer.step()
    return float(loss.data.mean())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', default=['./data/clone_detection/', './data/benign/', './data/malicious/'],
                        help='Paths to load training data for masked language model.')
    parser.add_argument('--total_longest_block', type=int, help='Length of longest block.', required=True)
    parser.add_argument('--d_word_vec', type=int, help='Dimension of token embeddings.', required=True)
    parser.add_argument('--n_layers', type=int, help='Number of layers', required=True)
    parser.add_argument('--n_head', type=int, help='Number of attention heads', required=True)
    parser.add_argument('--d_k', type=int, help='Dimension of key vectors', required=True)
    parser.add_argument('--d_v', type=int, help='Dimension of value vectors', required=True)
    parser.add_argument('--d_model', type=int, help='', required=True)
    parser.add_argument('--d_inner', type=int, help='Dimension of hidden layer of positionwise feedforward neural '
                                                    'network', required=True)
    parser.add_argument('--n_hidden', type=int, help='Dimension of hidden layer to predict tokens', required=True)
    parser.add_argument('--max_epochs', type=int, help='', required=True)
    parser.add_argument('--dropout', type=float, help='Dropout rate', required=True)

    paras = parser.parse_args()
    inputs, masks, outputs, token_map = load_masked_am_dataset(paras.paths)

    model_name = 'starplus_transformer'

    full_file_name = 'mask' + model_name + '_pretraining_model.pkl'
    block_level_file_name = 'block_level' + model_name + '.pkl'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total_longest_block = paras.total_longest_block
    r_token_map = {}
    for k, v in token_map.items():
        r_token_map[v] = k

    assert len(token_map) not in r_token_map
    assert len(token_map) + 1 not in r_token_map

    model = mask_star_satelitte_planet_pretraining_model(n_src_vocab=len(token_map), len_max_seq=total_longest_block, d_word_vec=paras.d_word_vec,
                                                         n_layers=paras.n_layers, n_head=paras.n_head, d_k=paras.d_k, d_v=paras.d_v,
                                                         d_model=paras.d_model, d_inner=paras.d_inner, n_hidden=paras.n_hidden,
                                                         n_out=len(token_map), token_map=token_map, dropout=paras.dropout).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total number of parameters:{}'.format(params))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    lb = preprocessing.LabelBinarizer()
    lb.fit(list(range(len(token_map))))

    epochs = paras.max_epochs



    load_full = False

    if load_full:
        checkpoint = torch.load(full_file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('load full success')

    n_sample = len(inputs)

    model.train()

    for epoch in range(epochs):
        touse_inputs = inputs
        touse_masks = masks
        touse_outputs = outputs
        losses = []
        batch_size = 5120
        bar = progressbar.ProgressBar(maxval=n_sample, prefix='epoch:' + str(epoch) + ' train').start()

        present = 0

        while present < n_sample:
            bar.update(present)
            x = touse_inputs[present:present + batch_size]
            m = touse_masks[present:present + batch_size]
            y = touse_outputs[present:present + batch_size]
            y = np.array(y).reshape(len(y) * 3)

            try:
                loss = train_one_step(x, m, device, model, criterion, optimizer)
                losses.append(loss)
                present += batch_size
            except Exception as e:
                print("exception:",e)
                batch_size = int(batch_size * 3 / 4)
                print('batch_size reduced to:', batch_size, 'with length of', len(x[-1]))
                continue

        print('[%d/%d] Loss: %.3f' % (epoch + 1, epochs, np.mean(losses)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, full_file_name)
        torch.save(model.encoder, block_level_file_name)


