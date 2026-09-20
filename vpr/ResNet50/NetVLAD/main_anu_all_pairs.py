import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning import losses, reducers
from os.path import join, exists, isfile, realpath, dirname
import loss_and_miner_utils as lmu
import utils
import clusters_nvlAgg_ as NVL
import argparse
import torch.nn.functional as F
import sys
sys.path.append('/data/anuradha/WACV25_Reviews/VPR/CosPlace-InfoNCE/')
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper
import common_functions as c_f
import gsv_cities as gsv
gsv_whole_set = gsv.WholeDatasetFromStruct()

parser = argparse.ArgumentParser(description='pytorch-NetVlad-gsv')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--num_clusters', type=int, default=16, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--dataPath', type=str, default='./init_data/', help='Path for centroid data.')
parser.add_argument('--arch', type=str, default='resnet50', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')        
parser.add_argument('--threads', type=int, default=20, help='Number of threads for each data loader to use')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--encoder_dim', type=int, default=1024, help='Dont use cuda')
opt = parser.parse_args() 


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
            
            q_neg_sim_mat = self.cosine_similarity(pos_embs, neg_embs)
            
            p_msk, n_msk = Upd_MultiSim.Mask_generator(q_pos_sim_mat, q_neg_sim_mat)
            
            
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
            del loss 
        return total_loss/self.embeddings.shape[0]



class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                init_path = True,
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                alpha=2, 
                beta=50, 
                base=0.5,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config
        self.init_path = init_path
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        self.alpha= alpha #nn.Parameter(torch.randn(1)) #
        self.beta=beta#nn.Parameter(torch.randn(1))
        self.base=base
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        #----- Upd_MultiSim from scratch
        self.upd_multisim = Upd_MultiSim
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config, init_path)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            
            Scratch_Msim = self.upd_multisim(descriptors, labels, miner_outputs, self.base, self.alpha, self.beta)
            loss = Scratch_Msim.Upd_MSim_Scratch()   
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else:
            print('===> No online miner is activated: Halting the code...')
            assert self.miner is not None

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        torch.cuda.empty_cache()
        
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []
        #print('===> Params:', 1+abs(self.alpha),'\t', 1+abs(self.beta))

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        return descriptors.detach().cpu()
    
    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 50, 100],
                                                gt=positives,
                                                print_results=True,
                                                dataset_name=val_set_name,
                                                faiss_gpu=self.faiss_gpu
                                                )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
            
            
if __name__ == '__main__':
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    if opt.mode.lower() == 'cluster':
        cuda = not opt.nocuda
        device = torch.device("cuda:1" if cuda else "cpu") 
        M = NVL.get_Cluster_NV(gsv_whole_set, opt, device)    
   
    elif opt.mode.lower() == 'train':
        datamodule = GSVCitiesDataModule(
             batch_size=100,
             img_per_place=4,
             min_img_per_place=4,
             shuffle_all=False, # shuffle all images or keep shuffling in-city only
             random_sample_from_each_place=True,
             image_size=(224, 224),
             num_workers=16,
             show_data_stats=True,
             val_set_names=['pitts30k_test']#, 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
        )

        model = VPRModel(
           #---- Encoder
           backbone_arch='resnet50',
           pretrained=True,
           layers_to_freeze=2,
           layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
         
        
           #---- Aggregator
           # agg_arch='CosPlace',
           # agg_config={'in_dim': 2048,
           #             'out_dim': 2048},
           # agg_arch='GeM',
           # agg_config={'p': 3},
        
           # agg_arch='ConvAP',
           # agg_config={'in_channels': 1024,
           #             'out_channels': 1024},

           # agg_arch='MixVPR',
           # agg_config={'in_channels' : 1024,
           #         'in_h' : 14,
           #         'in_w' : 14,
           #         'out_channels' : 1024,
           #         'mix_depth' : 4,
           #         'mlp_ratio' : 1,
           #         'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
        
           agg_arch='NetVLAD',
           agg_config={'num_clusters' : opt.num_clusters,
                       'dim' : 1024,
                       'vladv2' : opt.vladv2},
           init_path = join(opt.dataPath, 'centroids', opt.arch + '_' + gsv_whole_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5'),
        
        
           #---- Train hyperparameters
           lr=0.005, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
           optimizer='adamw', # sgd, adamw
           weight_decay=0,#.001, # 0.001 for sgd and 0 for adam,
           momentum=0.9,
           warmpup_steps=650,
           milestones=[5, 10, 15, 25, 45],
           lr_mult=0.3,

           #----- Loss functions
           # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
           # FastAPLoss, CircleLoss, SupConLoss,
           loss_name='MultiSimilarityLoss',
           miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
           miner_margin=0.1,
           alpha=1, 
           beta=50, 
           base=0.00,
           faiss_gpu=False
        )
    
       # model params saving using Pytorch Lightning
       # we save the best 3 models accoring to Recall@1 on pittsburg val
        checkpoint_cb = ModelCheckpoint(
           monitor='pitts30k_test/R1',
           filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_test/R1:.4f}]_R5[{pitts30k_test/R5:.4f}]',
           auto_insert_metric_name=False,
           save_weights_only=True,
           save_top_k=1,
           mode='max',
           save_last=False)

    #------------------
    #checkpoint_path = './LOGS_Upd_MultiSim/resnet50_epoch(01)_step(1251)_R1[0.8903]_R5[0.9462].ckpt'
    # we instanciate a trainer
        trainer = pl.Trainer(
           accelerator='gpu', devices=[0],
           default_root_dir=f'./ANU-All-Pairs/{model.encoder_arch}', # Tensorflow can be used to viz 

           num_sanity_val_steps=0, # runs a validation step before stating training
           precision=16, # we use half precision to reduce  memory usage
           max_epochs=40,
           check_val_every_n_epoch=1, # run validation every epoch
           callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
           reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
           log_every_n_steps=20,
        #resume_from_checkpoint=checkpoint_path
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
        )
 
    # we call the trainer, we give it the model and the datamodule
        trainer.fit(model=model, datamodule=datamodule)
