import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from torch_geometric.nn.models import SchNet, DimeNet, DimeNetPlusPlus
from mace import modules
from torch.nn import Parameter, Embedding
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
from torch import nn
import enum
import torch
from e3nn import o3
import abc
from mace.modules import MACE, ScaleShiftBlock
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)
from mace.tools.scatter import scatter_sum, scatter_mean
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Any, Callable, Dict, List, Optional, Type, Union

from utils import periodic_table_generator, ELEMENTS_NO
from mace.modules import ScaleShiftMACE


model_resolver = {
    "schnet": SchNet,
    "dimenet": DimeNet,
    "dimenetplusplus": DimeNetPlusPlus,
     "mace": ScaleShiftMACE
}

embedding_dimension_name_resolver = {
    "schnet": "hidden_channels",
    "dimenet": "hidden_channels",
    "dimenetplusplus": "hidden_channels",
    "mace": "num_elements"

}

def get_model_wrapper(all_models_params,
                 model_specs_dict, dataset_statistics=None):
    model_name = model_specs_dict["model_name"]
    if model_name in ["schnet", "dimenet", "dimenetplusplus"]:
        return PyGWrapper(all_models_params,model_specs_dict, dataset_statistics)
    elif model_name in ["mace","botnet"]:
        return MaceWrapper(all_models_params,model_specs_dict, dataset_statistics)



class ModelWrapperBase(torch.nn.Module,abc.ABC):
    """This class wraps SchNet class to handle different baselines with or without graph or periodic table feature matrix"""
    def __init__(self, all_models_params,
                 model_specs_dict, dataset_statistics=None):
        super().__init__()
        self.model_specs_dict = model_specs_dict
        self.model_name = model_specs_dict["model_name"]

        self.model_class = model_resolver[self.model_name]
        self.model_params = all_models_params[self.model_name]

        

        self.process_model_params(dataset_statistics)


        self.model = self.model_class(**self.model_params)
        elements_mlp = self.get_elements_mlp(model_specs_dict, all_models_params)
        if elements_mlp:
            self.set_elements_mlp(elements_mlp)
        

      

    def get_elements_mlp(self, model_specs_dict, all_models_params):
        if model_specs_dict["elements_mlp_type"]:
            mlp_layers_no =  model_specs_dict["elements_mlp_n_layers"]
            mlp_type =  model_specs_dict["elements_mlp_type"]
            periodic_data = periodic_table_generator()
            embedding_dim = self.get_embedding_dim(all_models_params)
            # This is to match the batch size 1 for our case
            elements_feats = periodic_data.x
            
            mlp_resolver = {
                "mlp": ElementsMLP,
                "glumlp": ElementsGLUMLP,
                "swiglumlp": ElementsSwiGLUMLP
            }
            elements_mlp = mlp_resolver[mlp_type](all_models_params["mlp_hidden_dim"], embedding_dim, elements_feats, mlp_layers_no)

            print(f"Chosen elements model is {mlp_type}")

            return elements_mlp
        else:
            return None
        
    @abc.abstractmethod
    def set_elements_mlp(self,elements_mlp):
            raise NotImplementedError
    
    @abc.abstractmethod
    def get_embedding_dim(self,all_models_params):
            raise NotImplementedError
    
    @abc.abstractmethod
    def process_model_params(self, dataset_statistics):
            raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, input_dict):
        raise NotImplementedError
    
    @abc.abstractmethod    
    def loss_fn(self, data):
        raise NotImplementedError
    
    @abc.abstractmethod   
    def predict(self,data):
        raise NotImplementedError
    
    @abc.abstractmethod   
    def get_targets(self,data):
        raise NotImplementedError
       

            


class MaceWrapper(ModelWrapperBase):
    def __init__(self, all_models_params,
                 model_specs_dict, dataset_statistics=None):
        self.elements_mlp = None
        super().__init__(all_models_params, model_specs_dict=model_specs_dict,dataset_statistics=dataset_statistics)
        self.embeddings = Embedding(100, self.get_embedding_dim(all_models_params), padding_idx=0)


    def forward(self, input_dict):
        if self.elements_mlp:
            node_attrs = self.elements_mlp(input_dict["data"].atomic_numbers.long())
            input_dict["data"].node_attrs = node_attrs
        else:
            #pass
            node_attrs = self.embeddings(input_dict["data"].atomic_numbers.long()-1)
            input_dict["data"].node_attrs = node_attrs
             
        return self.model.forward(input_dict["data"], training=input_dict["training"], compute_force=False)["energy"]
        
    def loss_fn(self, data):
        input_dict = {
            "data": data,
            "training": True
        }
        targets = data.energy
        return F.l1_loss(self.forward(input_dict),
                    targets)
       
    def predict(self,data):
        
        input_dict = {
            "data": data,
            "training": False
        }
        return self.forward(input_dict)
    def get_targets(self,data):
        return data.energy
    
    def set_elements_mlp(self,elements_mlp):
            self.elements_mlp = elements_mlp
    
    def get_embedding_dim(self,all_models_params):
            return ELEMENTS_NO
    
    def process_model_params(self, dataset_statistics):
            if self.model_name == "mace":
                self.model_params["interaction_cls"] = modules.interaction_classes[self.model_params["interaction_cls"]]
                self.model_params["num_elements"] = ELEMENTS_NO
                self.model_params["hidden_irreps"]=o3.Irreps(self.model_params["hidden_irreps"])
                self.model_params["gate"]=modules.gate_dict[self.model_params["gate"]]
                self.model_params["interaction_cls_first"] = modules.interaction_classes[self.model_params["interaction_cls_first"]]
                self.model_params["MLP_irreps"]=o3.Irreps(self.model_params["MLP_irreps"])
                self.model_params["radial_MLP"]=self.model_params["radial_MLP"]
                self.model_params["atomic_inter_scale"]=1.0
                self.model_params["atomic_numbers"]= torch.arange(1,ELEMENTS_NO+1)
                self.model_params["atomic_energies"]= torch.zeros((ELEMENTS_NO))
                self.model_params["avg_num_neighbors"] = dataset_statistics["avg_num_neighbors"]

                 

        

class PyGWrapper(ModelWrapperBase):
    def __init__(self, all_models_params,
                 model_specs_dict, dataset_statistics=None):
        super().__init__(all_models_params, model_specs_dict=model_specs_dict,dataset_statistics=dataset_statistics)

    def forward(self, input_dict):
        data = input_dict["data"]
        return self.model.forward(data.atomic_numbers.long(), data.pos, data.batch)
     
    def loss_fn(self, data):
            input_dict = {
                "data": data
            }
            targets= data.y
            return F.l1_loss(self.forward(input_dict).squeeze(dim=1),
                        targets)
       
    def predict(self,data):
        input_dict = {
            "data": data
        }
        return self.forward(input_dict).squeeze(dim=1)
    
    def get_targets(self,data):
        return data.y

    def set_elements_mlp(self,elements_mlp):
            if self.model_name == "schnet":
                self.model.embedding = elements_mlp
            elif self.model_name in ["dimenet", "dimenetplusplus"]:
                self.model.emb.emb = elements_mlp
    
    def get_embedding_dim(self,all_models_params):
        return all_models_params[self.model_name]["hidden_channels"]

    
    def process_model_params(self, dataset_statistics):
            pass
            

    


        


        



class ElementsMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, feat, num_layers):
        super(ElementsMLP, self).__init__()
        
        # Register the feature matrix as a buffer
        input_dim = feat.shape[1]
        self.register_buffer('feat', feat)
        
        # Define the MLP layers
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    def forward(self, z):
        # Pass the feature matrix through the MLP
        processed_feat = self.mlp(self.feat)
        # Select the relevant features based on indices z
        embeddings = processed_feat[z-1]
        return embeddings
    

class ElementsGLUMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, feat, num_layers):
        super(ElementsGLUMLP, self).__init__()
        
        # Register the feature matrix as a buffer
        input_dim = feat.shape[1]
        self.register_buffer('feat', feat)
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.final_layer_norm = nn.LayerNorm(output_dim)
    def forward(self, z):
        processed_feat = self.mlp(self.feat)
        processed_feat = self.final_layer_norm(processed_feat)
        embeddings = processed_feat[z-1]
        return embeddings


class ElementsSwiGLUMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, feat, num_layers):
        super(ElementsSwiGLUMLP, self).__init__()
        
        # Register the feature matrix as a buffer
        input_dim = feat.shape[1]
        self.register_buffer('feat', feat)
        
        layers = []
        if num_layers == 1:
            layers.append(SwiGLU(input_dim, output_dim))
        else:
            layers.append(SwiGLU(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(SwiGLU(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(SwiGLU(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.final_layer_norm = nn.LayerNorm(output_dim)
    def forward(self, z):
        processed_feat = self.mlp(self.feat)
        processed_feat = self.final_layer_norm(processed_feat)
        embeddings = processed_feat[z-1]
        return embeddings

class SwiGLU(nn.Module):
    
    def __init__(self, input_dim,output_dim) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, output_dim, bias=False)
        self.lin2 = nn.Linear(input_dim, output_dim, bias=False)
        self.lin3 = nn.Linear(output_dim, output_dim, bias=False)

    
    def forward(self, x):
        x1 = F.silu(self.lin1(x))
        x2 = self.lin2(x)
        
        
        return self.lin3(x1 * x2)