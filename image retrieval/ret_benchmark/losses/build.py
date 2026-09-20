# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from .multi_similarity_loss import MultiSimilarityLoss
from .anu_multi_similarity_loss import ANU_MultiSimilarityLoss
from .margin_loss import MarginLoss
from .triplet_loss import TripletLoss
from .anu_triplet_loss import ANU_TripletLoss

from .fastap_loss import FastAPLoss
from .anu_fastap_loss import ANU_FastAPLoss

from .npairs_loss import NPairsLoss
from .anu_npairs_loss import ANU_NPairsLoss

from .registry import LOSS


def build_loss(cfg):
    loss_name = cfg.LOSSES.NAME
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not registered in registry :{LOSS.keys()}'
    return LOSS[loss_name](cfg)
