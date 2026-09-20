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
from src import utils



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
        self.trip_loss = losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
        self.trip_miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard") #miners.MultiSimilarityMiner(epsilon=0.1)
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
        mined_pairs = self.trip_miner(descriptors, labels)
        #loss =  self.ms_loss(descriptors, labels, mined_pairs)
        loss = self.trip_loss(descriptors, labels, mined_pairs)
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
