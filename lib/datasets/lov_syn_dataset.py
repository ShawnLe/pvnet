import sys

sys.path.append('.')
sys.path.append('..')

import os
import torch
import torch.utils.data as data

from lib.utils import data_utils
from lib.datasets.linemod_dataset import LineModDatasetRealAug

class LovSynDataset(data.Dataset):
    """ 
        just a light wrapper of LovSynImageDB to become data.Dataset
        it just reads the indexes and puts to a dictionary
    """

    def __init__(self):

        self.dataset_name = 'lov_syn'
        self.lov_data =  data_utils.LovSynImageDB()


    def __len__(self):

        return self.lov_data.num_images


    def __getitem__(self, index):

        return self.lov_data.roidb[index]



class LovSynDatasetRealAug(data.Dataset):
    """
        this adds augmentation information to image and meta data
        it reads actual data and augments them
    """

    def __init__(self, imageDB):

        self._imageDB = imageDB
        self.data_prefix = ''

        self._imageDB_linemod = self.convert2LineMod()
        self.linemodAug = LineModDatasetRealAug(self._imageDB_linemod)

    def __len__(self):

        return self.linemodAug.__len__


    def __getitem__(self, index):

        return self.linemodAug.__getitem__[index]


    def convert2LineMod(self):
        """
        needs to provide
        imagedb[index]['rgb_pth']
        imagedb[index]['dpt_pth']
        """


        return self._imageDB
                
