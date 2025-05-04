import torch.nn.functional as F
import torch
import os.path as osp
import lightning as L
from lightning.pytorch.utilities import grad_norm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class LightningWrapper(L.LightningModule):
    def __init__(self, model, optimizer, scheduler, general_params,):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.targets = []
        self.predictions = []

        self.general_params = general_params
        self.scheduler = scheduler




    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data = batch

        loss = self.model.loss_fn(data)
        
        self.log("train_loss", loss.item())


        return loss
    
    def test_step(self, batch, batch_idx):
        pass
     

    def validation_step(self, batch, batch_idx):
        data = batch    
        pred,target = self.model.predict(data), self.model.get_targets(data)
        self.targets.append(target)
        self.predictions.append(pred)

     
    def on_validation_epoch_end(self):
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        predictions = torch.cat(self.predictions, dim=0).detach().cpu().numpy()
        mae_mean = mean_absolute_error(targets, predictions)
        mse_mean = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        self.log("val_mae_mean", mae_mean, sync_dist=True)
        self.log("val_mse_mean", mse_mean, sync_dist=True)
        self.log("val_r2_mean", r2, sync_dist=True)
        self.targets = []
        self.predictions = []
     

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

