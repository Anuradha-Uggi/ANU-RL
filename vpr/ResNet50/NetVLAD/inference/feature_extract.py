#!/usr/bin/env python



from __future__ import print_function
import sys
import pytorch_lightning as pl
import argparse
import configparser
import os
from os import makedirs
import random
from os.path import join, isfile, exists
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from datasets_original_bb import PlaceDataset
from tqdm.auto import tqdm
sys.path.append('./..')
from models import helper_inf as helper
import sys
sys.path.append('/data/anuradha/WACV25_Reviews/VPR/')
from MSLS_Val.MapillaryDataset import MSLS  


class VPRModel(pl.LightningModule):
    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=2,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, MS-NetVLAD, GeM
                agg_config={},
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config
        
        self.backbone = helper.get_backbone(self.encoder_arch, self.pretrained, self.layers_to_freeze, self.layers_to_crop)
        self.aggregator = helper.get_aggregator(self.agg_arch, self.agg_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)        
        return x

def Model(opt, config, checkpoint, device):
    model = VPRModel(
           backbone_arch='resnet50',
           pretrained=True,
           layers_to_freeze=3,
           layers_to_crop=[4],         
           agg_arch='NetVLAD',
           agg_config={'num_clusters' : opt.num_clusters,
                       'dim' : 1024,
                       'vladv2' : opt.vladv2})                 
    
    model.load_state_dict(checkpoint)#state_dict['state_dict'])
    model.eval()
    return model

def mixvpr_feature_extract(eval_set, model, device, opt, config, file_name):
    
    if not exists(opt.output_features_dir):
        makedirs(opt.output_features_dir)
    output_global_features_filename = join(opt.output_features_dir, file_name+'_feats_pool1.npy')
    pool_size = 16384#int(config['global_params']['num_pcs'])
    
    test_data_loader = DataLoader(dataset=eval_set, num_workers=int(config['global_params']['threads']),
                                  batch_size=int(config['feature_extract']['cacheBatchSize']),
                                  shuffle=False, pin_memory=False)
    model=model.to(device)
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        feat_pool = np.empty((len(eval_set), pool_size), dtype=np.float32) 
                       
        for iteration, (input_data, indices) in \
                enumerate(tqdm(test_data_loader, position=1, leave=False, desc='Test Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device); 
            image_encoding_pool = model.forward(input_data)
            feat_pool[indices_np, :] = image_encoding_pool.detach().cpu().numpy()
    np.save(output_global_features_filename, feat_pool)


def input_transform(resize=(224, 224)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Extract')
    parser.add_argument('--config_path', type=str, default='performance_bb.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data')    
    parser.add_argument('--query_file_path', type=str, default='/home/anuradha/Anni@A100/WACV25/Patch-NetVLAD/patchnetvlad/dataset_imagenames/tokyo247_imageNames_query.txt',
                        help='Full path (with extension) to a text file that stores the save location and name of all images in the dataset folder')                    
    parser.add_argument('--index_file_path', type=str, default='/home/anuradha/Anni@A100/WACV25/Patch-NetVLAD/patchnetvlad/dataset_imagenames/tokyo247_imageNames_index.txt',
                        help='Full path (with extension) to a text file that stores the save location and name of all images in the dataset folder')
    
    parser.add_argument('--dataset_root_dir', type=str, default='/data/anuradha/Used-Datasets-WACV-25/',
                        help='If the files in dataset_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--model_path', type=str, default='./triplet/bl/resnet50_epoch(10)_step(6886)_R1[0.8043]_R5[0.9102].ckpt')
    parser.add_argument('--arch', type=str, default='resnet50', 
                        help='basenetwork to use', choices=['resnet50','vgg16', 'alexnet'])   
    parser.add_argument('--output_features_dir', type=str, default='./triplet/bl/')   
    parser.add_argument('--msls', action='store_true', help='None')
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    parser.add_argument('--num_clusters', type=int, default=16, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    opt = parser.parse_args()
    print(opt)
    
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")    
    opt.device = device = torch.device("cuda:1" if cuda else "cpu")
    
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    
    model=Model(opt, config, checkpoint['state_dict'], device)
    
    if not opt.msls:
       print('===> Non-MSLS activated')
       qImgs = PlaceDataset(None, opt.query_file_path, opt.dataset_root_dir, None, config['feature_extract'])
       dbImgs = PlaceDataset(None, opt.index_file_path, opt.dataset_root_dir, None, config['feature_extract'])
    
       qfeats = mixvpr_feature_extract(qImgs, model, device, opt, config, 'qry')
       dbfeats = mixvpr_feature_extract(dbImgs, model, device, opt, config, 'db')
    
    elif opt.msls:
       print('===> MSLS activated')
       my_transform = input_transform()
       val_dataset = MSLS(my_transform, None)
       qImgs = MSLS(my_transform, 'qry')
       dbImgs = MSLS(my_transform, 'db')
       
       qfeats = mixvpr_feature_extract(qImgs, model, device, opt, config, 'qry')
       dbfeats = mixvpr_feature_extract(dbImgs, model, device, opt, config, 'db')     
    
if __name__ == "__main__":
    main()
        
