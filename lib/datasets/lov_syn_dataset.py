import os
import torch
import torch.utils.data as data


class lov_syn_dataset(data.Dataset):

    def __init__(self, path):
        self.dataset_name = 'lov_syn'



    def __len__(self):

        return 0


    def __getitem__(self, index):

        return None

