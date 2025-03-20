from model_wrapper import get_model_wrapper
import torch
import lightning as L
from torch_geometric.loader import DataLoader
import os.path as osp
from lightning.pytorch.callbacks import LearningRateMonitor
import sys
import os
from dataset import init_data
from lightning_trainer import LightningWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

from mace.data.atomic_data import get_data_loader
from mace import modules
from lightning.pytorch.strategies import DDPStrategy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import ELEMENTS_NO, parse_model_specs
torch.set_float32_matmul_precision('medium')


class Experiment:
    """Each experiment contains a set of general parameters which can be found in general.json file,
    a dataset, a set of baselines which is specified in the config files. The run function runs each
     of the baselines on the dataset and saves the results. """

    def __init__(self, model_params,
                 general_params,
                 dataset_config,
                 experiment_name,
                 runs_no=1,
                 save_directory="./results/"):
        self.experiment_name = experiment_name
        self.general_params = general_params
        self.model_params = model_params
        self.save_directory = save_directory
        self.runs_no = runs_no
        self.dataset_config = dataset_config
        self._registered_models = general_params["registered_models"]


    def run(self):

        for run_number in range(self.runs_no):
            print(f"Experiment's run number: {run_number}")
            print()
            for model_specs in self._registered_models:
                model_specs_dict = parse_model_specs(model_specs)
                model_name = model_specs_dict["model_name"]

                train_loader, test_loader, dataset_statistics = init_data(self.dataset_config, self.general_params,model_name)


                model = get_model_wrapper(deepcopy(self.model_params), model_specs_dict, dataset_statistics)


                optimizer = torch.optim.AdamW(model.parameters(), self.general_params[model_name]["lr"])
                scheduler = CosineAnnealingLR(optimizer, 
                                                T_max= self.general_params[model_name]["epochs"],
                                                )
                lightning_model = LightningWrapper(model,optimizer,scheduler, self.general_params)
                lr_monitor = LearningRateMonitor(logging_interval='step')
                trainer = L.Trainer(devices=self.general_params["device_no"], 
                                    accelerator=self.general_params["accelerator"], 
                                    default_root_dir= os.path.join(self.general_params["logs_dir"],self._get_save_string(model_specs, run_number)),
                                    log_every_n_steps=self.general_params["log_every_n_steps"],
                                    gradient_clip_val=self.general_params["max_grad_norm"],
                                    accumulate_grad_batches= self.general_params[model_name]["accumulate_grad_batches"],
                                    detect_anomaly=False,
                                    max_epochs = self.general_params[model_name]["epochs"],
                                    callbacks= [lr_monitor],
                                    strategy=DDPStrategy(find_unused_parameters=True),
                                    enable_checkpointing=True)  

                trainer.logger.log_hyperparams({
                    "train_dataset_size": len(train_loader.dataset),
                    "test_dataset_size": len(test_loader.dataset)
                }) 

                trainer.fit(lightning_model, train_loader, test_loader)


 
    def _get_save_string(self, model_name, run_number):
        return f"{self.experiment_name}_{model_name}_{run_number}"



