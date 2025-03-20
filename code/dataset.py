import sys
import os
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Batch, Data
import os.path as osp
from tqdm import tqdm
from matminer.datasets import load_dataset
import pandas as pd
from mace.data.atomic_data import AtomicData
from mace.data.neighborhood import get_neighborhood
import torch.distributed as dist
from mace.tools import (
    to_one_hot,
)
import numpy as np
Batch.__len__ = lambda self: len(self.y)
import lightning
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from torch_geometric.loader import DataLoader
from mace import modules
from utils import save_object,load_object, random_subset, ELEMENTS_NO



target_names = {
    "matbench_mp_gap": "gap pbe",
    "matbench_mp_e_form": "e_form"

}


class MaterialsBenchmark(Dataset):
    """Gets a root and and the dataset name and applies a filter on
    train dataset and another filter on the test dataset.
    This class manages the dataset for evaluation."""
    def __init__(self, root="/storage/home/hmm5834/scratch/iid-matbench-data/", dataset_name="matbench_mp_e_form",
                 train_ratio=0.8, transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 test_pre_filter=None,
                 data_interface="normal",
                 cutoff = 5.0, elements_no=94, scale_train_dataset=False, rgen=None):

        self._dataset = None
        self._train_test_threshold = None
        self.data_interface = data_interface
        self.rgen=rgen

        self.dataset_name = dataset_name

        self.train_ratio = train_ratio
        self.dataset_raw = None
        self.test_pre_filter = test_pre_filter
        self.scale_train_dataset=scale_train_dataset
        
        self.cutoff = cutoff
        self.elements_no = elements_no
        self.process_rank = int(os.environ.get("LOCAL_RANK", 0))

        
    
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = load_object(osp.join(self.processed_dir, self.processed_file_names[0]))
        return self._dataset

    def len(self):
        return len(self.dataset)

    @property
    def target_name(self):
        return target_names[self.dataset_name]

    @property
    def train_test_threshold(self):
        if self._train_test_threshold is None:
            self._train_test_threshold = load_object(osp.join(self.processed_dir, self.processed_file_names[1])).item()

        return self._train_test_threshold

    @property
    def raw_file_names(self):
        return [f"raw.pickle-{self.process_rank}"]

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}-{self.data_interface}-{self.process_rank}.pickle", f"train_test_threshold-{self.process_rank}.pickle"]

    def download(self):
        dataset_raw = load_dataset(self.dataset_name)
        permutation = torch.randperm(len(dataset_raw), generator=self.rgen).tolist()
        dataset_raw = dataset_raw.iloc[permutation].reset_index(drop=True)
        dataset_raw.to_pickle(osp.join(self.raw_dir, self.raw_file_names[0]))

    def process(self):
        dataset_raw = pd.read_pickle(osp.join(self.raw_dir, self.raw_file_names[0]))

        # train-test split
        raw_train_test_threshold = int(len(dataset_raw) * self.train_ratio)

        datalist_train = self._df_to_datalist(dataset_raw[:raw_train_test_threshold], self.pre_filter)
        datalist_test = self._df_to_datalist(dataset_raw[raw_train_test_threshold:], self.test_pre_filter)


        if self.scale_train_dataset:
            #scale training data like the filtered test set
            subset_len = int(len(datalist_train)*(len(dataset_raw[raw_train_test_threshold:])-len(datalist_test))/len(dataset_raw[raw_train_test_threshold:]))
            datalist_train = datalist_train[:subset_len]



        dataset = datalist_train + datalist_test
        self._dataset = dataset
        self._train_test_threshold = torch.tensor(len(datalist_train))
        save_object(self._train_test_threshold, osp.join(self.processed_dir, self.processed_file_names[1]))
        save_object(self._dataset, osp.join(self.processed_dir, self.processed_file_names[0]))
        

    def _df_to_datalist(self, df, df_filter=None):
        dataset = []
        for index, (_, row) in tqdm(enumerate(df.iterrows())):
            atomic_numbers = torch.tensor(row['structure'].atomic_numbers, dtype=torch.int)
            if self.data_interface == "normal":
                data = self._get_normal_format(row['structure'], row[self.target_name])
            elif self.data_interface == "mace":
                data = self._get_mace_format(row['structure'], row[self.target_name])

            if ((df_filter is None) or
                    df_filter(atomic_numbers)):
                dataset.append(data)
        return dataset
    

    def _get_normal_format(self,structure,target):

        pos = torch.tensor(structure.cart_coords).float()
        atomic_numbers = torch.tensor(structure.atomic_numbers, dtype=torch.int)
        return Data(pos=pos, y=target, atomic_numbers=atomic_numbers, atoms_no=len(atomic_numbers))

    
    def _get_mace_format(self, structure,target):
        positions = structure.cart_coords
        indices = np.array(structure.atomic_numbers)
        cell = structure.lattice.matrix  # shape (3, 3)
        pbc = structure.pbc

   
        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=positions, cutoff=self.cutoff, pbc=pbc, cell=cell
        )
        one_hot = to_one_hot(
            torch.tensor(indices-1, dtype=torch.long).unsqueeze(-1),
            num_classes=self.elements_no,
        )

        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        

        
        energy = (
            torch.tensor(target, dtype=torch.get_default_dtype())
            if target is not None
            else None
        )

        weight = torch.tensor(1)
        energy_weight = torch.tensor(1)
        forces_weight = torch.tensor(1)
        stress_weight = torch.tensor(1)
        virials_weight = torch.tensor(1)

        forces = (None)
        stress = (None)
        virials = (None)
        dipole = (None)
        charges = (None)
        atoms_no=torch.tensor(len(indices))

       

        return Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            node_attrs=one_hot,
            weight=weight,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            charges=charges,
            atoms_no=atoms_no,
            atomic_numbers = torch.tensor(indices, dtype=torch.get_default_dtype())
        )

    def get_train_test_split(self):
        return self.dataset[:self.train_test_threshold], self.dataset[self.train_test_threshold:]

    def get(self, idx):

        if idx >= len(self):
            raise IndexError("Index is larger than the dataset size")

        return self._dataset[idx]
    


def  init_data(dataset_config, general_params, model_name):

    if "mace" in model_name:
        data_interface = "mace"
    else:
        data_interface = "normal"

    dataset = MaterialsBenchmark(**dataset_config, data_interface=data_interface,
                                    cutoff=general_params["cutoff"], elements_no=ELEMENTS_NO, 
                                    rgen=torch.Generator().manual_seed(general_params["seed"]))
    print(f"len train dataset {dataset.train_test_threshold}")


    train_data, test_data = dataset.get_train_test_split()

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=general_params[model_name]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers = 5
    )   

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=general_params[model_name]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers = 5

    )
    dataset_statistics = {

    }

    if data_interface == "mace":
        dataset_statistics["avg_num_neighbors"] = modules.compute_avg_num_neighbors(train_loader)

    return train_loader, test_loader, dataset_statistics
