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


class Upd_MultiSim():
    def __init__(self, 
                 #--- Embeddings
                 embeddings=None,
                 labels=None,
                 miner_outputs=None,
                 loss_type=None,
                 base = 0.5,
                 alpha = 2,
                 beta = 50,
                 lambda_=1
                 ):
        super().__init__()
        self.cosine_similarity = DotProductSimilarity()
        self.embeddings = embeddings
        self.ref_emb = embeddings
        self.labels = labels
        self.indices_tuple = miner_outputs
        self.loss_type = loss_type
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.mat = self.cosine_similarity(self.embeddings, self.ref_emb)
        self.lambda_ = lambda_
    
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
    
    def Hardest_ANU_Pairs(sim_mat, msk, msk_type):   
        anu_rel = sim_mat[1:,] # Retaining all relationships for the query
        #print('********', sim_mat.shape, anu_rel.shape, msk_type)
        if msk_type == 'pos':
           extrm_vals, _ = anu_rel.min(dim=1, keepdim=True)
        elif msk_type == 'neg':
           extrm_vals, _ = anu_rel.max(dim=1, keepdim=True)
        msk_new = torch.where(anu_rel == extrm_vals, anu_rel, torch.zeros_like(anu_rel))
        msk_new = (msk_new>0).float()
        msk_new = torch.vstack([msk[0,:], msk_new])
        return msk_new
    
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
            
            q_neg_sim_mat = self.cosine_similarity(pos_embs, neg_embs)
            
            p_msk, n_msk = Upd_MultiSim.Mask_generator(q_pos_sim_mat, q_neg_sim_mat)
            
            rp, cp = q_pos_sim_mat.shape; rn, cn = q_neg_sim_mat.shape 
            
            if self.loss_type == 'hardest':
                if rp>1 and cp>0: p_msk = Upd_MultiSim.Hardest_ANU_Pairs(q_pos_sim_mat, p_msk, 'pos')
                if rn>1 and cn>0: n_msk = Upd_MultiSim.Hardest_ANU_Pairs(q_neg_sim_mat, n_msk, 'neg')
            elif self.loss_type == 'easiest': 
                if rp>1 and cp>0: p_msk = Upd_MultiSim.Easiest_ANU_Pairs(q_pos_sim_mat, p_msk, 'pos')
                if rn>1 and cn>0: n_msk = Upd_MultiSim.Easiest_ANU_Pairs(q_neg_sim_mat, n_msk, 'neg')
            
            mat_nest = torch.cat((q_pos_sim_mat, q_neg_sim_mat), dim=1)
            
            pos_exp = self.base-q_pos_sim_mat#*diag_sim_mat#q_pos_sim_mat_per_p
            neg_exp = q_neg_sim_mat-self.base
          
            pos_loss = (1.0 / self.alpha) * Upd_MultiSim.logsumexp(
                x=self.alpha * pos_exp, keep_mask=p_msk.bool(), add_one=True
            )
            neg_loss = (1.0 / self.beta) * Upd_MultiSim.logsumexp(
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
            
            #loss_q = pos_loss[0]+neg_loss[0]
            #loss_p2p = ((pos_loss[1:] + neg_loss[1:]).sum())#/len(pos_loss) 
            #loss = loss_q +  self.lambda_ * (loss_p2p)
            #total_loss = total_loss + loss
            del loss
            #loss = (pos_loss + neg_loss).sum() 
            #total_loss = total_loss + loss
            #del loss    
        return total_loss/self.embeddings.shape[0] 
        
        

@LOSS.register('anu_ms_loss')
class ANU_MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(ANU_MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        
        self.scale_pos = self.alpha = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = self.beta = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.loss_type = cfg.LOSSES.TYPE
        self.loss_fn = losses.MultiSimilarityLoss(alpha=self.scale_pos, beta=self.scale_neg, base=self.thresh, distance=DotProductSimilarity())
        self.miner = miners.MultiSimilarityMiner(epsilon=self.margin, distance=DotProductSimilarity()) 
        self.lambda_ = cfg.LOSSES.REG_LAMBDA
        self.anu_multisim = Upd_MultiSim
        
    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        miner_outputs = self.miner(feats, labels)
        ANU_Msim = self.anu_multisim(feats, labels, miner_outputs, self.loss_type, self.thresh, self.scale_pos, self.scale_neg, self.lambda_)
        loss = ANU_Msim.Upd_MSim_Scratch()
        
        return loss        
        
        
        
