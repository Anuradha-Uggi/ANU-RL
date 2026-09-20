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
    def __init__(self, cfg, normalize=False):
        super(NPairsLoss, self).__init__()
        self.normalize = normalize
        self.temperature = 0.07

    def custom_op(self, anchor, pos, negs):
        return anchor, pos, negs

    def forward(self, feats, labels):
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)

        B = feats.size(0)
        device = feats.device
        sim = torch.matmul(feats, feats.t()) / self.temperature

        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float('-inf'))

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)

        losses = []
        for i in range(B):
            pos_idx = torch.where(pos_mask[i])[0]
            neg_idx = torch.where(~pos_mask[i] & ~eye[i])[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            anchor = feats[i]
            pos = feats[pos_idx[0]]
            negs = feats[neg_idx]

            anchor, pos, negs = self.custom_op(anchor, pos, negs)

            pos_sim = torch.matmul(anchor, pos) / self.temperature
            neg_sim = torch.matmul(negs, anchor) / self.temperature

            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=0).unsqueeze(0)
            target = torch.zeros(1, dtype=torch.long, device=device)

            losses.append(F.cross_entropy(logits, target))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()
