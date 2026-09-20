import sys
import pytorch_lightning as pl
import argparse
import configparser
import os
import copy
from os import makedirs
import random
from torch.autograd import grad
from os.path import join, isfile, exists
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
#from torch.autograd.functional import hessian
from pyhessian import hessian

import matplotlib.pyplot as plt
sys.path.append('/data/anuradha/Anus-PhD/WACV25_Reviews/VPR/WACV25/Multi-Sim/CosPlace/')
from models import helper
import utils
sys.path.append('/data/anuradha/Anus-PhD/WACV25_Reviews/VPR/')#CosPlace-InfoNCE/')
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


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
        
        self.loss_fn = utils.get_loss('MultiSimilarityLoss')
        self.miner = utils.get_miner('MultiSimilarityMiner', 0.1) 
        
        self.agg_arch = agg_arch
        self.agg_config = agg_config
        
        self.backbone = helper.get_backbone(self.encoder_arch, self.pretrained, self.layers_to_freeze, self.layers_to_crop)
        self.aggregator = helper.get_aggregator(self.agg_arch, self.agg_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)        
        return x

def Model(checkpoint):
    model = VPRModel(
           backbone_arch='resnet50',
           pretrained=True,
           layers_to_freeze=3,
           layers_to_crop=[4],         
           agg_arch='CosPlace',
           agg_config={'in_dim': 1024,
                       'out_dim': 1024},)               
    
    model.load_state_dict(checkpoint) #state_dict['state_dict'])
    model.eval()
    return model


def compute_hessian_eigenvalues(model, loss):
    # Flatten all model parameters into one vector
    params = [p for p in model.parameters() if p.requires_grad]
    flat_params = torch.cat([p.view(-1) for p in params])
    grads = autograd.grad(loss, params, create_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

    H_size = flat_grads.numel()
    H = torch.zeros(H_size, H_size).to(flat_grads.device)

    for i in range(H_size):
        grad2rd = autograd.grad(flat_grads[i], params, retain_graph=True)
        grad2rd_flat = torch.cat([g.contiguous().view(-1) for g in grad2rd])
        H[i] = grad2rd_flat

    # Compute eigenvalues of the Hessian
    eigenvalues = torch.linalg.eigvalsh(H)  # Symmetric version (real-valued)
    return eigenvalues.cpu().detach().numpy()


def hessian_vector_product(model, loss, v):
    grads = grad(loss, model.parameters(), create_graph=True)
    grad_vec = parameters_to_vector(grads)
    dot = torch.dot(grad_vec, v)
    hvp = grad(dot, model.parameters(), retain_graph=True)
    return parameters_to_vector(hvp)


# This is a simple function, that will allow us to perturb the model paramters and get the result
def perturb_model(model, model_copy, ev1, ev2, alpha1, alpha2):
    #with torch.no_grad():
     cc=0
     '''for m_orig, m_copy in zip(model.parameters(), model_copy):
        if m_orig.requires_grad and m_copy.requires_grad:
           print('-----', m_orig.shape, m_copy.shape, ev1[cc].shape, ev2[cc].shape)
    '''
     with torch.no_grad():
        for m_orig, m_copy in zip(model.parameters(), model_copy):
           if m_orig.requires_grad and m_copy.requires_grad:
              #print('??????', m_orig.shape, m_copy.shape)  
              m_orig.copy_(m_copy + alpha1 * ev1[cc] + alpha2 *ev2[cc])
              cc += 1 
              #print(m_copy.shape)
     print('====== the end* ======')

def plot_3d_heatmap(X, Y, Z, name):
     # 3D heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='k', color='darkred', linewidth=0.2, antialiased=True)
    # Set view angle to match paper
    ax.view_init(elev=20, azim=60)
    # Axis labels using LaTeX style
    ax.set_xlabel('$\\epsilon_1$', labelpad=10)
    ax.set_ylabel('$\\epsilon_2$', labelpad=10)
    ax.set_zlabel('Loss', labelpad=10)
    ax.zaxis.label.set_rotation(90)
    ax.grid(True)
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(Z.min(), Z.max())
    #plt.title('Loss Landscape GeM+Ours')
    plt.tight_layout()
    plt.savefig(name)#)  
    plt.close()
    

def plot_2d_contour(X, Y, Z, file_name):
    # 2D Contours
    
    levels = np.linspace(Z.min(), Z.max(), 15)
    
    plt.figure(figsize=(6,5))

    # filled contour
    contour = plt.contourf(X, Y, Z, 
                          levels=levels, 
                          cmap='RdYlBu_r', 
                          extend='both')
    
    # contour lines
    lines = plt.contour(X, Y, Z, 
                levels=levels, 
                colors='black', 
                linewidths=0.3, 
                alpha=0.8)
    
    # labels for contour lines
    plt.clabel(lines, inline=True, fontsize=8)
    
    plt.xlabel(r'$\epsilon_1$', fontsize=12)
    plt.ylabel(r'$\epsilon_2$', fontsize=12)

    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)

    # Colorbar matching style
    cbar = plt.colorbar(contour, label='Loss', shrink=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=9)

    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig(file_name)#)#, dpi=300)
    plt.close() 

def loss_surface(model, top_eigenvectors, images, labels):
    alpha_range = np.linspace(-0.05, 0.05, 21)
    loss_surface = np.zeros((21, 21)) 
    model_copy = [p.clone() for p in model.parameters()]
    ev1, ev2 = top_eigenvectors[0], top_eigenvectors[1]
    
    #with torch.no_grad():
    for i, alpha1 in enumerate(alpha_range):
       for j, alpha2 in enumerate(alpha_range):
          model_perb = perturb_model(model, model_copy, ev1, ev2, alpha1, alpha2)
          descriptors = model.forward(images)
          miner_outputs = model.miner(descriptors, labels)
          loss = model.loss_fn(descriptors, labels, miner_outputs) 
          loss_surface[i, j] = loss.item()
    # Restore original parameters
    with torch.no_grad():
       for p, orig in zip(model.parameters(), model_copy):
           p.copy_(orig)    
    #print('norm of eigen values:', ev1, ev2)#torch.norm(ev1), torch.norm(ev2))  # Should not be zero
    #print('norm of loss surface:', loss_surface.min(), loss_surface.max()) 
    X, Y = np.meshgrid(alpha_range, alpha_range)
    Z = loss_surface
    
    #plot_3d_heatmap(X, Y, Z, 'BL_CosPlace_60.png')
    plot_2d_contour(X, Y, Z, 'BL_CosPlace_contour.png')

def main():
    parser = argparse.ArgumentParser(description='Hessian')
    #parser.add_argument('--config_path', type=str, default='./inference/performance_bb.ini',
    #                    help='File name (with extension) to an ini file that stores most of the configuration data')    
    parser.add_argument('--bl_model_path', type=str, default='/data/anuradha/Anus-PhD/WACV25_Reviews/VPR/WACV25/Multi-Sim/CosPlace/LOGS-MSim-BL/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(19)_step(12520)_R1[0.8908]_R5[0.9450].ckpt')
    parser.add_argument('--gsv_cities_path', type=str, default='/data/anuradha/Anus-PhD/Gsv-Cities/')
    parser.add_argument('--arch', type=str, default='resnet50', 
                        help='basenetwork to use', choices=['resnet50','vgg16', 'alexnet'])   
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    opt = parser.parse_args()
    print(opt)
    
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    #configfile = opt.config_path
    #assert os.path.isfile(configfile)
    #config = configparser.ConfigParser()
    #config.read(configfile)
    
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")    
    opt.device = device = torch.device("cuda" if cuda else "cpu")
    
    datamodule = GSVCitiesDataModule(
        batch_size=60,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=8,
        show_data_stats=True,
        val_set_names=['pitts30k_test'], # pitts30k_val, pitts30k_test, msls_val
    )
    
    
    #ckpts = #os.listdir(opt.model_path)
    #ckpts_upd = #os.listdir(opt.upd_model_path)
    
    checkpoint = torch.load(opt.bl_model_path, map_location=lambda storage, loc: storage)
    
    model=Model(checkpoint['state_dict']).to(device)
    
    #for name, p in model.named_parameters():
    #    print(name, p.requires_grad)
    
    dm = datamodule
    
    train_data_loader = dm.train_dataloader()
    
    places, train_labels = next(iter(train_data_loader))
    BS, N, ch, h, w = places.shape
    images = places.view(BS*N, ch, h, w); labels = train_labels.view(-1)
    images = images.to(device); 
    
    x = images; y = labels;
    #descriptors = model.forward(images)
    #miner_outputs = model.miner(descriptors, labels)
    #loss = model.loss_fn(descriptors, labels, miner_outputs) 
    
    #for name, p in model.named_parameters():
    #   print(name, p.grad is None)
    #print(hello)
    #params = [p for p in model.parameters() if p.requires_grad]
    #params_vec = parameters_to_vector(params)
    #params_devec = vector_to_parameters(params_vec, params)
    #print(params_devec)
    
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    
    '''def loss_fun():
        descriptors = model.forward(images)
        miner_outputs = model.miner(descriptors, labels)
        loss = model.loss_fn(descriptors, labels, miner_outputs) 
        return loss'''
    #for mm in model.parameters():
    #    print('models:', mm.shape)    
    #H = hessian(loss_fun, params_vec)

    # Optional: convert to NumPy, compute eigenvalues
    #eigenvalues = torch.linalg.eigvalsh(H).cpu().numpy()'''
    
    hessian_comp = hessian(model, model.loss_fn, (x, y))
    eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=2)
    loss_surface(model, top_eigenvectors, images, labels)
    
    #density, grids = hessian_comp.density()
    
    '''plt.semilogy(np.array(grids), np.array(density) + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    plt.savefig('MSim_eigen_density.png')'''
    
    
    
    print('==== Hessian Compute Done ====')
    
    
if __name__ == "__main__":
    main()
        
