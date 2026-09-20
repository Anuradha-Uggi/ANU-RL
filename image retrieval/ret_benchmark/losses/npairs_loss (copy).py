# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from ret_benchmark.losses.registry import LOSS
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity


@LOSS.register('npairs_loss')
class NPairsLoss(nn.Module):
    def __init__(self, cfg):
        super(NPairsLoss, self).__init__()
        self.loss_fn = losses.NTXentLoss(temperature=0.07) #NPairsLoss()

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        loss = self.loss_fn(feats, labels)
        '''batch_size = feats.size(0)
        #sim_mat = torch.matmul(feats, torch.t(feats))
        #loss = list()
        loss = 0
        target = torch.tensor([0], device=feats.device) 
        for i in range(batch_size):
            pos_idx = (labels == labels[i]).nonzero(as_tuple=False).flatten()
            pos_idx = pos_idx[pos_idx != i][0]
     
            #positive = pos_idx[0]
            neg_idx = (labels != labels[i]).nonzero(as_tuple=False).flatten()
            
            anchor = feats[i]
            positive = feats[pos_idx]
            negatives = feats[neg_idx]
            
            pos_score = anchor @ positive
            neg_scores_q = anchor @ negatives.T; #neg_scores_p = positive @ negatives.T
            
            logits_qpn = torch.cat([pos_score.unsqueeze(0), neg_scores_q])
            #logits_pqn = torch.cat([pos_score.unsqueeze(0), neg_scores_p])
            
            loss_qpn = F.cross_entropy(logits_qpn.unsqueeze(0), target)
            #loss_pqn = F.cross_entropy(logits_pqn.unsqueeze(0), target)
            
            loss_t = loss_qpn#+loss_pqn
            loss += loss_t
            
        loss = loss / batch_size #sum(loss) / batch_size #'''
        return loss
