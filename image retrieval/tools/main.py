# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import argparse
import torch
from os import makedirs
import random
from os.path import join, isfile, exists

import sys
sys.path.append('/data/anuradha/Anus-PhD/WACV25_Reviews/IR/From_Sindhu/')
from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.engine.trainer import do_train
from ret_benchmark.losses import build_loss
from ret_benchmark.modeling import build_model
from ret_benchmark.solver import build_lr_scheduler, build_optimizer
from ret_benchmark.utils.logger import setup_logger
from ret_benchmark.utils.checkpoint import Checkpointer


def train(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    criterion = build_loss(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0
    if not exists(cfg.SAVE_DIR):
        makedirs(cfg.SAVE_DIR) 
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = Checkpointer(model, optimizer, scheduler, cfg.SAVE_DIR)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        device,
        checkpoint_period,
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
    train(cfg)

