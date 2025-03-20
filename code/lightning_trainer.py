import torch.nn.functional as F
import torch
import os.path as osp
import lightning as L
from lightning.pytorch.utilities import grad_norm

class LightningWrapper(L.LightningModule):
    def __init__(self, model, optimizer, scheduler, general_params,):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.val_mae_list = []
        self.val_mae_per_atom_list = []
        self.general_params = general_params
        self.scheduler = scheduler




    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data = batch

        loss = self.model.loss_fn(data)
        
        self.log("train_loss", loss.item())


        return loss
    
    def test_step(self, batch, batch_idx):
        data = batch    
        pred, target= self.model.predict(data), self.model.get_targets(data)

        mae_per_atom = (pred.view(-1) - target).abs() / data.atoms_no
        mae = (pred.view(-1) - target).abs() / len(data)

        self.log("mae_per_atom", mae_per_atom, sync_dist=True)
        self.log("mae", mae, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        data = batch    
        pred,target = self.model.predict(data), self.model.get_targets(data)
        mae_per_atom = (pred.view(-1) - target).abs() / data.atoms_no
        mae = (pred.view(-1) - target).abs()
        self.val_mae_list.append(mae)
        self.val_mae_per_atom_list.append(mae_per_atom)

    def on_validation_epoch_end(self):
        mae_mean = torch.cat(self.val_mae_list, dim=0).mean().item()
        self.val_mae_list = []
        self.log("val_mae_mean", mae_mean, sync_dist=True)
        mae_per_atom_mean = torch.cat(self.val_mae_per_atom_list, dim=0).mean().item()
        self.val_mae_per_atom_list = []
        self.log("val_mae_per_atom", mae_per_atom_mean, sync_dist=True)

    def configure_optimizers(self):
       
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'epoch',  # or 'step'
                'frequency': 1
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

