import numpy as np
import torch
from torch.autograd import Variable
import argparse
import yaml
from easydict import EasyDict
import os
import h5py
import sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
from fewshot.utils import model_util, data_util, com_util

def main(params):
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

    # data
    data_loader = data_util.get(params, False)
    
    # model
    model = model_util.get_model(params)

    # load model
    save_dir = com_util.get_save_dir(params)
    if params.save_iter != -1:
        modelfile = com_util.get_assigned_file(save_dir, params.save_iter)
    else:
        modelfile = com_util.get_best_file(save_dir)
    model.load_state_dict(torch.load(modelfile)['state_dict'])
    model = model.feature
    model.eval()

    # save
    split = params.split
    if params.save_iter != -1:
        outfile = os.path.join(save_dir.replace("ckpts", "feats"), split + "_" + str(params.save_iter)+ ".hdf5")
    else:
        outfile = os.path.join(save_dir.replace("ckpts", "feats"), split + ".hdf5")
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_feats(model, data_loader, outfile)


def save_feats(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x,y) in enumerate(data_loader):
        #if i % 10 == 0:
        #    print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--config', default='cfgs/baseline/miniImagenet.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config['common'])
    params.update(config['test'])

    main(params)
