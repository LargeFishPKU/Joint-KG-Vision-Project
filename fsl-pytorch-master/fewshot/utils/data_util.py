import os
import os.path as osp
from ..data import SimpleDataManager, SetDataManager

num_classes = {
    'omniglot': 4112,
    'cross_char': 1597,
    'cross': 1000,
    'CUB': 200,
    'miniImagenet': 100
}

def get(params, train=True):
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    params.num_classes = num_classes[params.dataset]

    if train:
        return get_train_data(params, image_size)
    else:
        return get_test_data(params, image_size)


def get_train_data(params, image_size):
    if params.dataset == 'cross':
        base_file = 'dataset/miniImagenet/all.json'
        val_file   = 'dataset/CUB/val.json'
    elif params.dataset == 'cross_char':
        base_file = 'dataset/omniglot/noLatin.json'
        val_file   = 'dataset/emnist/val.json'
    else:
        base_file = osp.join('dataset', params.dataset, 'base.json')
        val_file  = osp.join('dataset', params.dataset, 'val.json')

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size = 16)
        base_loader = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr = SimpleDataManager(image_size, batch_size = 64)
        val_loader = val_datamgr.get_data_loader( val_file, aug = False)
    else:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way))
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, n_episode=params.n_episode)
        if params.use_text_vector:
            train_few_shot_params['text_vector_path'] = params.text_vector_path
        base_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

        test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot)
        if params.use_text_vector:
            test_few_shot_params['text_vector_path'] = params.text_vector_path
        val_datamgr = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader( val_file, aug = False)

    return base_loader, val_loader


def get_test_data(params, image_size):
    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = 'dataset/miniImagenet/all.json'
        else:
            loadfile = 'dataset/miniImagenet/' + split +'.json'
    elif params.dataset == 'cross_char':
        if split == 'base':
            loadfile = 'dataset/omniglot/noLatin.json'
        else:
            loadfile = 'dataset/emnist/' + split +'.json'
    else:
        loadfile = osp.join('dataset', params.dataset, split + '.json')

    if params.method in ['maml', 'maml_approx', 'protonet', 'protonet_joint']:
        few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
        if params.use_text_vector:
            few_shot_params['text_vector_path'] = params.text_vector_path
        datamgr = SetDataManager(image_size, n_episode = params.iter_num, n_query = 15 , **few_shot_params)
        data_loader = datamgr.get_data_loader(loadfile, aug = False)
    else:
        datamgr = SimpleDataManager(image_size, batch_size = 64)
        data_loader = datamgr.get_data_loader(loadfile, aug = False)

    return data_loader
