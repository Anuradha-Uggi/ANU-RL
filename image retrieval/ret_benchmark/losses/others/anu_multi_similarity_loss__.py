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
#import loss_and_miner_utils as lmu 


@LOSS.register('anu_ms_loss')
class ANU_MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(ANU_MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.loss_fn = losses.MultiSimilarityLoss(alpha=self.scale_pos, beta=self.scale_neg, base=self.thresh, distance=DotProductSimilarity())
        self.miner = miners.MultiSimilarityMiner(epsilon=self.margin, distance=DotProductSimilarity())

    def convert_to_triplets(self, indices_tuple, labels, ref_labels=None, t_per_anchor=100):
        if len(indices_tuple) == 3:
            return indices_tuple
        else:
            a1, p, a2, n = indices_tuple
            p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
            return a1[p_idx], p[p_idx], n[n_idx] 
     
    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        
        miner_outputs = self.miner(feats, labels)
        pairs2triplets = self.convert_to_triplets(miner_outputs, labels)
        anc, pos, neg = pairs2triplets[0], pairs2triplets[1], pairs2triplets[2]
        miner_outputs_apn = [anc, pos, neg]
        miner_outputs_pan = [pos, anc, neg]
        loss_apn = self.loss_fn(feats, labels, miner_outputs_apn)
        loss_pan = self.loss_fn(feats, labels, miner_outputs_pan)
        return loss_apn + loss_pan
