
import sys

sys.path.append('.')
sys.path.append('..')

from lib.utils import data_utils
from lib.datasets.lov_syn_dataset import LovSynDataset, LovSynDatasetRealAug

# dataset
lovDat = LovSynDataset()
augLovDat = LovSynDatasetRealAug(lovDat)

# dataloader

