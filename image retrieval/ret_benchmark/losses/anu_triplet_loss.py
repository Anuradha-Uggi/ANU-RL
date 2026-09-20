# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ret_benchmark.losses.registry import LOSS
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import torch.nn.functional as F
import common_functions as c_f


@LOSS.register('anu_triplet_loss')
class ANU_TripletLoss(nn.Module):
    def __init__(self, cfg):
        super(ANU_TripletLoss, self).__init__()
        self.margin = 0.1
        
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=False, smooth_loss=False, triplets_per_anchor='all')
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="semihard")
        
    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        miner_outputs = self.miner(feats, labels)
        anc, pos, neg = miner_outputs[0], miner_outputs[1], miner_outputs[2]
            
        miner_outputs_apn = [anc, pos, neg]
        miner_outputs_pan = [pos, anc, neg]
            
        loss_apn = self.loss_fn(feats, labels, miner_outputs_apn)       
        loss_pan = self.loss_fn(feats, labels, miner_outputs_pan)
            
        loss = loss_apn + loss_pan
        
        return loss        
        
        
        
