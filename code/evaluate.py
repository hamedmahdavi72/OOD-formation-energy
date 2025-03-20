import sys
import os
import argparse
import os.path as osp
import torch
from dataset import MaterialsBenchmark
from functools import partial
from experiment import Experiment

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import get_dataset_config,get_experiment_name, load_json
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, choices=["matbench_mp_gap", "matbench_mp_e_form"],
                    default="matbench_mp_e_form")
parser.add_argument("--configs_directory", type=str, default="./experiments_configs/")
parser.add_argument("--results_directory", type=str, default="./results/")

parser.add_argument("--datasets_directory", type=str, default="data")
parser.add_argument('--excluded_elements_list', nargs='+', type=int, help=" You determine the list of excluded elements ")
parser.add_argument("--experiment_type", type=str,
                     choices=["iid", "ood-list", "ood-list-train-iid", "ood-list-train-iid-scaled"],
                    default="iid")

# parser.add_argument('--registered_models', nargs='+', type=str, help=" You determine the list of registered models ")


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    model_params = load_json(osp.join(args.configs_directory, f"model.json"))
    general_params = load_json(osp.join(args.configs_directory, "general.json"))

    # if args.registered_models:
    #     general_params["registered_models"] = args.registered_models
    dataset_config = get_dataset_config(args.datasets_directory, args.dataset_name, args.experiment_type,
                          excluded_elements_list=args.excluded_elements_list)
    print(f"exlcuded elements list: {args.excluded_elements_list}")
    print(f"experiment type is {args.experiment_type}")
    experiment_name = get_experiment_name(args.experiment_type, excluded_elements_list=args.excluded_elements_list)

  
    experiment = Experiment(model_params, general_params, dataset_config, experiment_name,save_directory=args.results_directory)
    experiment.run()
