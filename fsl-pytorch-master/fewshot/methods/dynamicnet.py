import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .meta_template import MetaTemplate

class DynamicNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(DynamicNet, self).__init__(model_func, n_way, n_support)
        self.classifier = Classifier(1600)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        y_support = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support)))
        y_oh = one_hot(y_support, self.n_way)
        y_oh = Variable(y_oh.cuda())
        y_oh = y_oh.view(self.n_way, self.n_support, -1)
        scores = self.classifier(z_support, y_oh, z_query)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)


class Classifier(nn.Module):
    def __init__(self, nFeat):
        super(Classifier, self).__init__()
        self.nFeat = nFeat
        self.wnLayerFavg = LinearDiag(nFeat)
        self.bias = nn.Parameter(
            torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(10),
            requires_grad=True)

    def get_weights(self, features_train, labels_train):
        _, num_train_examples, num_channels = features_train.size()
        nKnovel = labels_train.size(2)
        weight_novel_avg = features_train.mean(1)
        #weight_novel = self.wnLayerFavg(weight_novel_avg.view(-1, num_channels))
        weight_novel = weight_novel_avg.view(nKnovel, num_channels)
        return weight_novel

    def apply_weights(self, features, cls_weights):
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12).view(-1, self.nFeat)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)
        #cls_scores = self.scale_cls * torch.mm(features, cls_weights.transpose(0,1)) + self.bias.view(1, 1)
        cls_scores = torch.mm(features, cls_weights.transpose(0,1)) 
        return cls_scores

    def forward(self, featrues_train, labels_train, features_test):
        cls_weights = self.get_weights(featrues_train, labels_train)
        cls_scores = self.apply_weights(features_test, cls_weights)
        return cls_scores

class LinearDiag(nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1).cuda()
