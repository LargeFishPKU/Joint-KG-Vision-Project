import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .meta_template import MetaTemplate


class DN4(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(DN4, self).__init__(model_func, n_way, n_support)
        self.imgtoclass = ImgtoClass_Metric(neighbor_k=3)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False, is_flatten=False):
        z_support, z_query = self.parse_feature(x, is_feature, is_flatten)
        z_query = z_query.contiguous().view(-1, *z_query.size()[2:])
        S = []
        for i in range(len(z_support)):
            b, c, h, w = z_support[i].size()
            z_support_sam = z_support[i].permute(1, 0, 2, 3).contiguous().view(c, -1)
            S.append(z_support_sam)
        output = self.imgtoclass(z_query, S)
        return output

    def set_forward_loss(self, x):
        output = self.set_forward(x)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        return self.loss_fn(output, y_query)


class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, input1, input2):
        B, C, h, w = input1.size()
        sim_lst = []

        for i in range(B):
            query_sam = input1[i]
            query_sam = query_sam.view(C, -1)
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()

            for j in range(len(input2)):
                support_set_sam = input2[j]
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
                support_set_sam = support_set_sam / support_set_sam_norm
                # cosine similarity between a query sample and a support
                # category
                innerproduct_matrix = query_sam@support_set_sam

                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(topk_value)

            sim_lst.append(inner_sim)

        sim_lst = torch.cat(sim_lst, 0)

        return sim_lst

    def forward(self, x1, x2):
        sim_lst = self.cal_cosinesimilarity(x1, x2)
        return sim_lst
