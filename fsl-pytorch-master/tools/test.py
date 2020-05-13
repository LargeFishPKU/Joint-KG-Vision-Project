import torch
import json
import os
import glob
import random
import time
import argparse
import yaml
import numpy as np
from easydict import EasyDict
import sys
import os.path as osp
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
from fewshot.data import feature_loader
from fewshot.utils import model_util, data_util,com_util

def main(params):
    acc_all = []

    print('dataset: {}, method: {}, model: {}'.format(params.dataset, params.method, params.model))

    # load model
    model = model_util.get_model(params, False)

    save_dir = com_util.get_save_dir(params)
    if params.save_iter != -1:
        modelfile = com_util.get_assigned_file(save_dir, params.save_iter)
    else:
        modelfile = com_util.get_best_file(save_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        # import pdb; pdb.set_trace()
        model.load_state_dict(tmp['state_dict'], strict=False)

    split = params.split
    iter_num = params.iter_num
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split

    # test
    if params.method in ['maml', 'maml_approx', 'protonet', 'protonet_joint']:
        novel_loader = data_util.get(params, False)
        model.eval()
        acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
    else:
        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        novel_file = os.path.join(save_dir.replace("ckpts", "feats"), split_str + ".hdf5")
        cl_data_file = feature_loader.init_loader(novel_file)
        ## 5 way 5 shot default
        for i in range(iter_num):
            acc = feat_eval(cl_data_file, model, n_query=15, adaptation=params.adaptation, **few_shot_params)
            acc_all.append(acc)
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num)))

    # record
    with open(osp.join(save_dir, 'record.txt'), 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++'] :
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' % (timestamp,exp_setting,acc_str))


def feat_eval(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query)])
    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query)
    acc = np.mean(pred == y) * 100
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--config', default='cfgs/baseline/miniImagenet.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config['common'])
    params.update(config['test'])

    main(params)
