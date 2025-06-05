import numpy as np
import os

from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, X_num):
        self.X_num = X_num

    def __getitem__(self, index):
        return self.X_num[index]

    def __len__(self):
        return self.X_num.shape[0]

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

