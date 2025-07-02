import torch
from torch.utils.data import  DataLoader
import random
import itertools
def build_dataloader( batch_size=1, shuffle=True, custom_dataset=None, collate_fn=None):
    """
    Builds a PyTorch DataLoader for the dataset where each question will have all permutations
    of its options.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the dataset before batching.
    :return: PyTorch DataLoader with permuted options.
    """
    if custom_dataset == None:
        print("Dataset cannot be None")
        return  
    else:
        dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader