# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import datetime
import time

import numpy as np
import torch
import argparse

'''import os
from collections import OrderedDict
import pretrainedmodels

import torch
from torch.nn.modules import Sequential

import sys
sys.path.append('/home/anuradha/MyResearch@Sindhu/WACV-ANUs-RL/IR/research-ms-loss/ret_benchmark/modeling/')
from backbone import build_backbone
from heads import build_head'''

from ret_benchmark.data.evaluations import RetMetric
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.freeze_bn import set_bn_eval
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.modeling import build_model
from ret_benchmark.utils.logger import setup_logger

def do_test(
        cfg,
        model,
        val_loader,
        device,
        arguments,
        logger
):
    logger.info("Start Validation")
    meters = MetricLogger(delimiter="  ")

    #start_training_time = time.time()
    #end = time.time()
    model.eval()
    logger.info('Validation')
    labels = val_loader.dataset.label_list
    labels = np.array([int(k) for k in labels])
    feats = feat_extractor(model, val_loader, logger=logger)

    ret_metric = RetMetric(feats=feats, labels=labels)
    recall_ks = [1, 5, 10]
    recall_curr = [ret_metric.recall_k(recall_ks[0]), ret_metric.recall_k(recall_ks[1]), ret_metric.recall_k(recall_ks[2])]

    logger.info(f'Recall@1: {recall_curr[0]:.5f}')
    logger.info(f'Recall@5: {recall_curr[1]:.5f}')
    logger.info(f'Recall@10: {recall_curr[2]:.5f}')
    
    #with open('./Experiments-SOP-Training-Data/128-dim/MSimLoss/output_anu_ms_loss/run_4/recalls.txt', 'w') as f:
    with open(cfg.SAVE_DIR, 'w') as f:
        for ind, re in enumerate(recall_curr):
           f.write(f"Recall: MSim: R@{recall_ks[ind]}: {re}\n" )
        #f.write(f"Recall: MSim: R@5: {recall_curr[1]}\n" )
        #f.write(f"Recall: MSim: R@10: {recall_curr[2]}" )
    f.close()    

    
def test(cfg):
    logger = setup_logger(name='Test', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    val_loader = build_data(cfg, is_train=False)

    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    do_test(
        cfg,
        model,
        val_loader,
        device,
        arguments,
        logger
    )


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file',
        default=None,
        type=str)
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command line"
    )     
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)   # <-- important
    test(cfg)   
