#!/bin/bash

#ood_list
python code/evaluate.py --experiment_type ood_list --excluded_elements_list 51 11 34 44 65 39 89 35 
python code/evaluate.py --experiment_type ood_list --excluded_elements_list 92 48 19 22 77 68 52 20 
python code/evaluate.py --experiment_type ood_list --excluded_elements_list 70 69 93 38 70 67 65 21 23 39
python code/evaluate.py --experiment_type ood_list --excluded_elements_list 14 53 75 68 49 71 73 52 15 17

#ood_list_train_iid
python code/evaluate.py --experiment_type ood_list_train_iid --excluded_elements_list 51 11 34 44 65 39 89 35
python code/evaluate.py --experiment_type ood_list_train_iid --excluded_elements_list 92 48 19 22 77 68 52 20
python code/evaluate.py --experiment_type ood_list_train_iid --excluded_elements_list 70 69 93 38 70 67 65 21 23 39
python code/evaluate.py --experiment_type ood_list_train_iid --excluded_elements_list 14 53 75 68 49 71 73 52 15 17

#ood_list_train_iid_scaled
python code/evaluate.py --experiment_type ood_list_train_iid_scaled --excluded_elements_list 51 11 34 44 65 39 89 35
python code/evaluate.py --experiment_type ood_list_train_iid_scaled --excluded_elements_list 92 48 19 22 77 68 52 20
python code/evaluate.py --experiment_type ood_list_train_iid_scaled --excluded_elements_list 70 69 93 38 70 67 65 21 23 39
python code/evaluate.py --experiment_type ood_list_train_iid_scaled --excluded_elements_list 14 53 75 68 49 71 73 52 15 17