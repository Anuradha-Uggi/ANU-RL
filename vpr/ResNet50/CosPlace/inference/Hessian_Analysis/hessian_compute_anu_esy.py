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
from pytorch_metric_learning.distances import CosineSimilarity

import matplotlib.pyplot as plt
sys.path.append('/data/anuradha/Anus-PhD/WACV25_Reviews/VPR/WACV25/Multi-Sim/CosPlace/')
from models import helper
import utils
import common_functions as c_f
import loss_and_miner_utils as lmu
sys.path.append('/data/anuradha/Anus-PhD/WACV25_Reviews/VPR/')#CosPlace-InfoNCE/')
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
       x = x.masked_fill(~keep_mask, c_f.neg_inf(x.dtype))
    if add_one:
       zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(dim)
       x = torch.cat([x, zeros], dim=dim)
    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
       output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        #print(output); print(hi)
        #output = output.sum()
    return output
    
def Mask_generator(q_pos_sim_mat, q_neg_sim_mat):
    p_msk = 1-torch.eye(q_pos_sim_mat.shape[0]).to('cuda') 
    n_msk = torch.ones(q_neg_sim_mat.shape).to('cuda') 
    return p_msk, n_msk

class ANU_MSim(nn.Module):
    def __init__(self, 
                 #--- Embeddings
                 #embeddings=None,
                 #labels=None,
                 #miner_outputs=None,
                 alpha=1, 
                 beta=50, 
                 base=0.00,
                 ):
        super().__init__()
        self.cosine_similarity = CosineSimilarity()
        #self.embeddings = embeddings
        #self.ref_emb = embeddings
        #self.labels = labels
        #self.indices_tuple = miner_outputs
        self.base = base
        self.alpha = alpha
        self.beta = beta
        #self.mat = self.cosine_similarity(self.embeddings, self.ref_emb)
    
    def Easiest_ANU_Pairs(sim_mat, msk, msk_type):   
        #anu_rel = sim_mat[1:,] # Retaining all relationships for the query
        #print('********', sim_mat.shape, anu_rel.shape, msk_type)
        if msk_type == 'pos':
           anu_rel_ = sim_mat-torch.eye(sim_mat.shape[0]).cuda()
           anu_rel = anu_rel_[1:,]
           extrm_vals, _ = anu_rel.max(dim=1, keepdim=True)
        elif msk_type == 'neg':
           anu_rel = sim_mat[1:,]
           extrm_vals, _ = anu_rel.min(dim=1, keepdim=True)
        msk_new = torch.where(anu_rel == extrm_vals, anu_rel, torch.zeros_like(anu_rel))
        msk_new = (msk_new>0).float()
        msk_new = torch.vstack([msk[0,:], msk_new])
        del anu_rel
        return msk_new
        
    def forward(self, embeddings, labels, miner_outputs):
        self.embeddings = embeddings
        self.ref_emb = embeddings
        self.labels = labels
        self.indices_tuple = miner_outputs
        self.mat = self.cosine_similarity(self.embeddings, self.ref_emb)
        
        a1, p, a2, n = self.indices_tuple
        pos_mask, neg_mask = torch.zeros_like(self.mat), torch.zeros_like(self.mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        total_loss = 0
        for q in range(self.embeddings.shape[0]):
            pos_inds = torch.where(pos_mask[q]==1) 
            neg_inds = torch.where(neg_mask[q]==1)
            q_emb =  self.embeddings[q]             
            pos_embs = torch.cat((self.embeddings[q].unsqueeze(0), self.embeddings[pos_inds]))  
            neg_embs = self.embeddings[neg_inds]  
            
            q_pos_sim_mat = self.cosine_similarity(pos_embs, pos_embs)
            
            q_neg_sim_mat = self.cosine_similarity(pos_embs, neg_embs)
            
            p_msk, n_msk = Mask_generator(q_pos_sim_mat, q_neg_sim_mat)
            
            rp, cp = q_pos_sim_mat.shape; rn, cn = q_neg_sim_mat.shape 
            if rp>1 and cp>0: p_msk = ANU_MSim.Easiest_ANU_Pairs(q_pos_sim_mat, p_msk, 'pos')
            if rn>1 and cn>0: n_msk = ANU_MSim.Easiest_ANU_Pairs(q_neg_sim_mat, n_msk, 'neg')
            
            mat_nest = torch.cat((q_pos_sim_mat, q_neg_sim_mat), dim=1)
            
            pos_exp = self.base-q_pos_sim_mat #*diag_sim_mat#q_pos_sim_mat_per_p
            neg_exp = q_neg_sim_mat-self.base
          
            pos_loss = (1.0 / self.alpha) * logsumexp(
                x=self.alpha * pos_exp, keep_mask=p_msk.bool(), add_one=True
            )
            neg_loss = (1.0 / self.beta) * logsumexp(
                x=self.beta * neg_exp, keep_mask=n_msk.bool(), add_one=True
            )
             
            '''pos_loss = pos_loss.unsqueeze(1)
            neg_loss = neg_loss.unsqueeze(1) 
             
            loss_dict = {
                "loss": {
                    "losses": pos_loss + neg_loss,
                    "indices": c_f.torch_arange_from_size(mat_nest),
                    "reduction_type": "element",
                }
            }
            reducer = reducers.MeanReducer()
            loss = reducer(loss_dict, self.embeddings, self.labels) '''
            loss = (pos_loss + neg_loss).sum() 
            total_loss = total_loss + loss
            del loss  
        return total_loss/self.embeddings.shape[0]


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
        
        self.miner = utils.get_miner('MultiSimilarityMiner', 0.1) 
        self.loss_fn = ANU_MSim() #losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
        
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


def plot_3d_heatmap(X, Y, Z, file_name):
    print("======= 3D Heatmap ========")
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
    plt.tight_layout()
    plt.savefig(file_name)#)  
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
    
    print("======= Plotting =========")
    X, Y = np.meshgrid(alpha_range, alpha_range)
    Z = loss_surface
    
    #plot_3d_heatmap(X, Y, Z, 'Anu_Esy_CosPlace_60.png')
    plot_2d_contour(X, Y, Z, 'ANU_Esy_CosPlace_contour.png')


def main():
    parser = argparse.ArgumentParser(description='Hessian')
    #parser.add_argument('--config_path', type=str, default='./inference/performance_bb.ini',
    #                    help='File name (with extension) to an ini file that stores most of the configuration data')    
    parser.add_argument('--anu_all_model_path', type=str, default='/data/anuradha/Anus-PhD/WACV25_Reviews/VPR/WACV25/Multi-Sim/CosPlace/ANU-Easiest-Pairs/resnet50/lightning_logs/version_0/checkpoints/resnet50_epoch(23)_step(15024)_R1[0.8859]_R5[0.9416].ckpt')
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
    
    checkpoint = torch.load(opt.anu_all_model_path, map_location=lambda storage, loc: storage)
    
    model=Model(checkpoint['state_dict']).to(device)
    
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
    
    #print('loss checks:', loss.item()); print(hiiik)
    
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
     
    #output = model.forward(x)
    #miner_outputs = model.miner(output,y)    
    #loss = model.loss_fn(output, y, miner_outputs).forward()
    
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
        
