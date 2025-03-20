import numpy
from itertools import chain
import torch
from torch_geometric.data import Data
import numpy as np
import json
import pickle
import threading
import os.path as osp
from functools import partial
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Create a global lock
ELEMENTS_NO = 94




"""This module provides a set of general util functions"""

def parse_model_specs(model_specs):
    model_specs_split = model_specs.split("-")
    model_specs_dict = {
        "model_name": model_specs_split[0]
    }
    if len(model_specs_split) == 3 and (model_specs_split[1] in ["mlp","glumlp", "swiglumlp"]) and (model_specs_split[2].isdigit()):
        model_specs_dict["elements_mlp_type"] = model_specs_split[1]
        model_specs_dict["elements_mlp_n_layers"] = int(model_specs_split[2])
    elif len(model_specs_split) == 1:
        model_specs_dict["elements_mlp_type"] = None
        model_specs_dict["elements_mlp_n_layers"] = None
    else:
        raise ValueError("model_specs string has an invalid value")
    return model_specs_dict

def save_object(obj, filename):

   
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_object(filename):

        with open(filename, 'rb') as file:
            return pickle.load(file)



def load_dict(address):

        with open(address, 'rb') as handle:
            out_dict = pickle.load(handle)
            return {int(k): v for k, v in out_dict.items()}


def save_dict(address, dict_struct):
        with open(address, 'wb') as handle:
            pickle.dump(dict_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(address):
    with open(address) as f:
        return json.load(f)


def split_list(lst, filter):
    pos_lst = []
    neg_lst = []
    for x in lst:
        if filter(x):
            pos_lst.append(x)
        else:
            neg_lst.append(x)
    return pos_lst, neg_lst


def random_subset(data, subset_size):
    if subset_size < 1:
        subset_size = int((len(data) * subset_size))

    idx = torch.randperm(len(data))

    if isinstance(data,torch.Tensor):
        return data[idx[:subset_size]]
    else:
        return [data[i] for i in idx[:subset_size].tolist()]


def periodic_table_generator(remove_horizontal=False):
    periodic_edges = [[(1, 3), (3, 11), (11, 19)],
                      [(4, 12), (12, 20)],
                      [(3, 4), (11, 12)],
                      [(2, 10)],
                      [(i, i + 18) for i in chain(range(13, 19), range(19, 37), range(37, 40))],
                      [(i, i + 8) for i in range(5, 11)],
                      [(i, i + 32) for i in chain(range(40, 55), range(55, 63))],

                      [(i, i + 1) for i in
                       chain(range(5, 10), range(13, 18), range(19, 36), range(37, 54), range(55, 86), range(87, 94))],
                      ]

    edges = []

    for edge_list in periodic_edges:
        edges.extend(edge_list)

    if remove_horizontal:
        edges = [edge for edge in edges if edge[1] > edge[0] + 1]

    edges = numpy.array(edges).transpose()

    # idx = preset.elements['atomic_number'] <= 54
    # relevant_elements = preset.elements[idx]
    #
    # relevant_elements = relevant_elements.dropna(axis=1)
    #
    # re_feats = relevant_elements.loc[:, relevant_elements.columns != 'atomic_number']
    # re_feats = (re_feats - re_feats.min()) / (
    #         re_feats.max() - re_feats.min())
    # np.save('periodic_feats.npy', re_feats)
    # np.save('periodic_edges.npy', edges)
    feats = np.load('periodic_feats.npy')
    feats = torch.tensor(feats).type(torch.FloatTensor)
    edges = torch.tensor(np.load('periodic_edges.npy')).long() - 1

    return Data(feats, edges)






def not_contains_specific_elements_list(data_atomic_numbers, checklist_atomic_numbers):
    return all([not_contains_specific_element(data_atomic_numbers, atomic_no) for atomic_no in checklist_atomic_numbers])


def contains_specific_elements_list(data_atomic_numbers, checklist_atomic_numbers):
    return any([contains_specific_element(data_atomic_numbers, atomic_no) for atomic_no in checklist_atomic_numbers])


def not_contains_specific_element(atomic_numbers, check_atomic_no):
    return torch.all(atomic_numbers != check_atomic_no)


def contains_specific_element(atomic_numbers, check_atomic_no):
    return torch.any(atomic_numbers == check_atomic_no)



def get_list_string(excluded_elements_list):
    return "_".join(str(atomic_no) for atomic_no in excluded_elements_list)

def get_dataset_config(datasets_directory, dataset_name, experiment_type, excluded_elements_list=None):
    if experiment_type == "iid":
        root = osp.join(datasets_directory, f"{dataset_name}_{experiment_type}")
        return {
            "root": root,
            "dataset_name": dataset_name
        }
        

    elif experiment_type == "ood-list":
        root = osp.join(datasets_directory,
                        f"{dataset_name}_{experiment_type}_{get_list_string(excluded_elements_list)}")
        pre_filter = partial(not_contains_specific_elements_list, checklist_atomic_numbers=excluded_elements_list)
        test_pre_filter = partial(contains_specific_elements_list, checklist_atomic_numbers=excluded_elements_list)
        return {
            "root": root,
            "dataset_name": dataset_name,
            "pre_filter": pre_filter,
            "test_pre_filter": test_pre_filter
        }
       
    elif experiment_type == "ood-list-train-iid":
        root = osp.join(datasets_directory,
                        f"{dataset_name}_{experiment_type}_{get_list_string(excluded_elements_list)}")
        test_pre_filter = partial(contains_specific_elements_list, checklist_atomic_numbers=excluded_elements_list)

        return {
            "root": root,
            "dataset_name": dataset_name,
            "test_pre_filter": test_pre_filter
        }
    elif experiment_type == "ood-list-train-iid-scaled":
        root = osp.join(datasets_directory,
                        f"{dataset_name}_{experiment_type}_{get_list_string(excluded_elements_list)}")
        test_pre_filter = partial(contains_specific_elements_list, checklist_atomic_numbers=excluded_elements_list)

        return {
            "root": root,
            "dataset_name": dataset_name,
            "test_pre_filter": test_pre_filter,
            "scale_train_dataset": True
        }


def get_experiment_name(experiment_type, excluded_elements_list=None):
    if experiment_type == "iid":
        return f"Experiment-{experiment_type}"
    elif experiment_type in ["ood-list", "ood-list-train-iid","ood-list-train-iid-scaled"] :
        return f"Experiment-{get_list_string(excluded_elements_list)}/{experiment_type}"
    else:
        raise ValueError("experiment type is not valid")
    


def load_tensorboard_logs(log_dir):

    # Initialize the EventAccumulator to load the logs
    event_acc = EventAccumulator(log_dir)
    
    # Load the events from the file
    event_acc.Reload()

    # Create a dictionary to store the scalar data
    log_data = {}

    # Get all scalar tags (e.g., loss, accuracy, etc.)
    scalar_tags = event_acc.Tags().get('scalars', [])

    # Iterate over the scalar tags and extract data
    for tag in scalar_tags:
        scalar_events = event_acc.Scalars(tag)
        log_data[tag] = [(event.wall_time, event.step, event.value) for event in scalar_events]

    return log_data


if __name__ == '__main__':
    data = periodic_table_generator()
    print(data)
    print(data.num_edges)
    print(data.num_nodes)
    print(np.argwhere(np.isnan(data.x.detach().cpu().numpy())))


