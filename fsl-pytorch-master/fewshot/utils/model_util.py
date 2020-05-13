import os
import os.path as osp
import torch
from ..models import backbone
from ..methods import BaselineTrain, BaselineFinetune, ProtoNet, MatchingNet, RelationNet, MAML, DN4, DynamicNet, ProtoNet_joint
from .com_util import *

model_dict = dict(
    Conv4 = backbone.Conv4,
    Conv4S = backbone.Conv4S,
    Conv4NP = backbone.Conv4NP,
    Conv6 = backbone.Conv6,
    ResNet10 = backbone.ResNet10,
    ResNet18 = backbone.ResNet18,
    ResNet34 = backbone.ResNet34,
    ResNet50 = backbone.ResNet50,
    ResNet101 = backbone.ResNet101
)

def get_model(params, train=True):
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if train and params.method in ['baseline', 'baseline++'] :
        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

    if params.method == 'baseline':
        if train:
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        else:
            model = BaselineFinetune(model_dict[params.model], **few_shot_params)
    elif params.method == 'baseline++':
        if train:
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type = 'dist')
        else:
            model = BaselineFinetune(model_dict[params.model], loss_type = 'dist', **few_shot_params)

    if params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **few_shot_params)
    elif params.method == 'protonet_joint':
        few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot,
                                    text_vector_dimension = params.text_vector_dimension)
        model = ProtoNet_joint(model_dict[params.model], **few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(model_dict[params.model], **few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model](flatten = False)
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model = RelationNet(feature_model, loss_type = loss_type, **few_shot_params)
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[params.model], approx = (params.method == 'maml_approx'), **few_shot_params)
        if params.dataset in ['omniglot', 'cross_char']:
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
        if not train:
            if params.adaptation:
                model.task_update_num = 100
    elif params.method == 'dn4':
        model = DN4(model_dict[params.model], **few_shot_params)
    elif params.method == 'dynamicnet':
        model = DynamicNet(model_dict[params.model], **few_shot_params)

    model = model.cuda()
    return model

def load_model(params, modelfile):
    model = get_model(params)
    model.load_state_dict(torch.load(modelfile))
    model.cuda()
    return model.feature

def resume(params, model, optimizer):
    start_epoch = 0
    if params.resume:
        resume_file = get_resume_file(params.save_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        print('load {}'.format(params.save_dir))
    elif params.warmup:
        baseline_save_dir = '%s/ckpts/%s/%s_%s' %('exp', params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_save_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_save_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state_dict']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    return model, optimizer, start_epoch
