import sys

sys.path.append('.')
sys.path.append('..')

import os
import torch
import torch.utils.data as data

from lib.utils import data_utils


class lov_syn_dataset(data.Dataset):
    """ 
        just a light wrapper of LovSynImageDB to become data.Dataset
    """

    def __init__(self, path):

        self.dataset_name = 'lov_syn'
        self.lov_data =  data_utils.LovSynImageDB()


    def __len__(self):

        return self.lov_data.num_images


    def __getitem__(self, index):

        return self.lov_data.roidb[index]



class lov_syn_DatasetRealAug(data.Dataset):
    """
        this adds augmentation information to image and meta data
    """

    def __init__(self, imageDB):

        self._imageDB = imageDB
        self._aug_db = do_augment()
        self.data_prefix = ''

    def __len__(self):

        return self._imageDB.lov_data.num_images


    def __getitem__(self, index):

        rgb_path = os.path.join(self.data_prefix, self._imageDB.lov_data.roidb[index]['image'])
        mask_path = os.path.join(self.data_prefix, self._imageDB.lov_data.roidb[index]['depth'])  # check

        

        return _aug_db[index]


    def do_augment(self):



        return aug_db