# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import pytorch_lightning as L
from pytorch_metric_learning import losses, miners
import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity
from src import utils
import common_functions as c_f
import loss_and_miner_utils as lmu

class Upd_MultiSim():
    def __init__(self, 
                 #--- Embeddings
                 embeddings=None,
                 labels=None,
                 miner_outputs=None,
                 base = 0.5,
                 alpha = 2,
                 beta = 50
                 ):
        super().__init__()
        self.cosine_similarity = CosineSimilarity()
        self.embeddings = embeddings
        self.ref_emb = embeddings
        self.labels = labels
        self.indices_tuple = miner_outputs
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.mat = self.cosine_similarity(self.embeddings, self.ref_emb)
    
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
    
    def MSim_Scratch(self):
        a1, p, a2, n = self.indices_tuple
        pos_mask, neg_mask = torch.zeros_like(self.mat), torch.zeros_like(self.mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        pos_exp = self.base-self.mat
        neg_exp = self.mat-self.base

        pos_loss = (1.0 / self.alpha) * Upd_MultiSim.logsumexp(
            x=self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * Upd_MultiSim.logsumexp(
            x=self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        
        loss_dict = {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(self.mat),
                "reduction_type": "element",
            }
        }
        reducer = reducers.MeanReducer()
        loss = reducer(loss_dict, self.embeddings, self.labels)   
        return loss  
    
    def Upd_MSim_Scratch(self):
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
            #q_pos_upper_tri = torch.triu(q_pos_sim_mat, diagonal=1)  # Upper triangular without diagonal
            #q_neg_lower_tri = torch.tril(q_pos_sim_mat, diagonal=-1) # Lower triangular without diagonal
            #diag_sim_mat = torch.eye(q_pos_sim_mat.shape[0]).to('cuda') 
            # Combine and flatten
            #q_pos_sim_mat_per_p = torch.cat((q_pos_upper_tri, q_neg_lower_tri))
            #q_pos_sim_mat = q_pos_sim_mat - diag_sim_mat  
            q_neg_sim_mat = self.cosine_similarity(pos_embs, neg_embs)
            
            p_msk, n_msk = Upd_MultiSim.Mask_generator(q_pos_sim_mat, q_neg_sim_mat)
            
            
            mat_nest = torch.cat((q_pos_sim_mat, q_neg_sim_mat), dim=1)
            
            pos_exp = self.base-q_pos_sim_mat#*diag_sim_mat#q_pos_sim_mat_per_p
            neg_exp = q_neg_sim_mat-self.base
          
            pos_loss = (1.0 / self.alpha) * Upd_MultiSim.logsumexp(
                x=self.alpha * pos_exp, keep_mask=p_msk.bool(), add_one=True
            )
            #pos_loss = (1.0 / self.alpha) * torch.log(torch.sum(torch.exp(self.alpha * pos_exp), dim=1))
            neg_loss = (1.0 / self.beta) * Upd_MultiSim.logsumexp(
                x=self.beta * neg_exp, keep_mask=n_msk.bool(), add_one=True
            )
            #neg_loss = (1.0 / self.beta) * torch.log(torch.sum(torch.exp(self.beta * neg_exp), dim=1)+1) 
             
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
            loss = reducer(loss_dict, self.embeddings, self.labels)  
            total_loss = total_loss + loss'''
            loss_q = pos_loss[0]+neg_loss[0]
            loss_p2p = ((pos_loss[1:] + neg_loss[1:]).sum())#/len(pos_loss) 
            
            loss = 1.0 * (loss_q) #+ 0.1 * (loss_p2p)
            total_loss = total_loss + loss
            
            del loss 
        return total_loss/self.embeddings.shape[0] 


class BoQModel(L.LightningModule):
    def __init__(
            self, 
            backbone, 
            aggregator,
            lr=1e-4,
            lr_mul=0.1,
            weight_decay=1e-3,
            warmup_epochs=10,
            milestones=[10, 20],
            silent=False,
            alpha=2,
            beta=50, 
            base=0.5,
        ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.lr = lr
        self.lr_mul = lr_mul
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.silent = silent # disable console output
        self.alpha=alpha
        self.beta=beta 
        self.base=base
        # init loss function and miner
        self.ms_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1)
        # Upd Msim
        self.upd_multisim = Upd_MultiSim
    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns
        
    def configure_optimizers(self):
        optimizer_params = [
            {"params": self.backbone.parameters(),   "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mul
        )    
        return [optimizer], [scheduler]
    
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmup_epochs:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    @torch.compiler.disable()
    def compute_loss(self, descriptors, labels):
        mined_pairs = self.ms_miner(descriptors, labels)
        #loss =  self.ms_loss(descriptors, labels, mined_pairs)
        Scratch_Msim = self.upd_multisim(descriptors, labels, mined_pairs, self.base, self.alpha, self.beta)
        #loss = Scratch_Msim.MSim_Scratch()
        loss = Scratch_Msim.Upd_MSim_Scratch()
        return loss
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images.shape is (P, K, C, H, W) with P: number of places, K: number of views per place
        # labels.shape is (P, K)
        images = images.flatten(0, 1) # P*K, C, H, W 
        labels = labels.flatten() # P*K
        
        # forward pass
        descriptors, attentions = self(images)
        # compute loss
        loss = self.compute_loss(descriptors, labels)
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss 

    def on_train_epoch_end(self):
        # reload the dataframes to shuffle in-city
        # this is faster than reloading the entire dataloader
        #self.trainer.train_dataloader.dataset._refresh_dataframes()
        ds = self.trainer.train_dataloader.dataset
        if hasattr(ds, "_refresh_dataframes"):
           ds._refresh_dataframes()
           
    def on_validation_epoch_start(self):
        # we init an empty dictionary to store the descriptors for each dataloader
        self.validation_outputs = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        descriptors, attentions = self(images)
        descriptors = descriptors.detach().cpu()#.numpy()
        
        if dataloader_idx not in self.validation_outputs:
            # keep in mind that we might have multiple validation dataloaders
            # initialize an empty list for this dataloader, then append the descriptors
            self.validation_outputs[dataloader_idx] = []
            
        # save the descriptors to compute the recall@k at the end of the validation epoch
        self.validation_outputs[dataloader_idx].append(descriptors)

    def on_validation_epoch_end(self):
        # get the validation dataloaders        
        val_dataloaders = self.trainer.val_dataloaders
        recalls = {} # one dict for each validation set
        for dataloader_idx, descriptors_list in self.validation_outputs.items():
            descriptors = torch.cat(descriptors_list, dim=0)
            dataset = val_dataloaders[dataloader_idx].dataset

            if self.trainer.fast_dev_run:
                # skip the recall computation for fast dev runs
                if dataloader_idx == 0:
                    print("\nFast dev run: skipping recall@k computation\n")
            else:
                # we will use the descriptors, the number of references, number of queries, and the ground truth
                # NOTE: make sure these are available in the dataset object and ARE IN THE RIGHT ORDER.
                # meaning that the first `num_references` descriptors are reference images and the rest are query images
                recalls_dict = utils.compute_recall_performance(
                        descriptors, 
                        dataset.num_references,
                        dataset.num_queries,
                        dataset.ground_truth,
                        k_values=[1, 5, 10, 15],
                )
                recalls_log = {
                    f"{dataset.dataset_name}/R@1": recalls_dict[1],
                    f"{dataset.dataset_name}/R@5": recalls_dict[5],
                }
                recalls[dataset.dataset_name] = recalls_dict
                
                # add to the logger but not the progress bar 
                # we will display the results below
                self.log_dict(recalls_log, prog_bar=False, logger=True)
        
        if recalls and not self.silent:
            utils.display_recall_performance(
                list(recalls.values()), 
                list(recalls.keys()),
            )
        self.validation_outputs.clear()
