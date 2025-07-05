import numpy as np
# 1. Load CIFAR10 and extract the eigenvalues of 2 classes

import torch
import torch.nn as nn
from dask import datasets

from config import NNAnalysisParameters
from evaluation import analyze_flip_threshold
from train import train_c1c2cov
import scipy

def create_random_data():
    # Create 10 random orthogonal matrices

    num_of_matrices = 10
    d = 32**2
    matrices = []
    for i in range(num_of_matrices):
        A = scipy.stats.ortho_group.rvs(d)
        matrices.append(A)

    matrices = np.array(matrices)
    # Save the data
    np.save('random_matrices.npy', matrices)





if __name__ == '__main__':

    # First - Create the data
    # create_random_data()
    # matrices = np.load('random_matrices.npy')

    # # Train The random data:
    # cfg = NNAnalysisParameters()
    # img_size = {'CIFAR10': 32}
    # ### Flipping Test For Gaussians ###
    # configs = [('FC', 'CIFAR10', False)]
    # c1c2_list = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    #              (0, 6), (0, 7), (0, 8), (0, 9)]
    # # c1c2_list = [(0, 3)]
    # for i in range(len(configs)):
    #     cfg.model_type, cfg.dataset_type, cfg.normalize_all_db = configs[i]
    #     cfg.img_width = img_size[cfg.dataset_type] ** 2
    #     for c1c2 in c1c2_list:
    #         train_loss, test_loss = train_c1c2cov(cfg, c1c2, show=True)


    cfg = NNAnalysisParameters()
    img_size = {'CIFAR10': 32}

    ### Flipping Test For Gaussians ###
    configs = [('FC', 'CIFAR10', False)]
    c1c2_list = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (0, 6), (0, 7), (0, 8), (0, 9)]
    # c1c2_list = [(0, 3)]
    for i in range(len(configs)):
        cfg.model_type, cfg.dataset_type, cfg.normalize_all_db = configs[i]
        cfg.img_width = img_size[cfg.dataset_type] ** 2
        analyze_flip_threshold(cfg, c1c2_list, flip_eigenvectors=True, flip_eigenvalues=False, load_data=False)
        analyze_flip_threshold(cfg, c1c2_list, flip_eigenvectors=False, flip_eigenvalues=True, load_data=False)


