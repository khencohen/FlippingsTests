import numpy as np
import torch

torch.random.manual_seed(42)

class NNAnalysisParameters:

    img_width = 28
    input_channels = 1
    hidden_size = 2048
    output_size = 2
    num_of_layers = 3       # For the FC model

    N = 28 ** 2
    num_samples = 1000
    dataset_type = 'RANDOM_NORMAL'
    epochs = 20
    lr = 1e-3
    weight_decay = 1e-2
    batch_size = 128

    train_test_split_ratio = 0.8

    model_type = 'FC'

    nn_header = True
    linear_model = False
    limit_train_data = None
    num_batch_samples = 5
    max_num_of_db_samples = np.inf
    normalize_all_db = False
    same_eig = False
    same_eigVec = False
    rotate_eigenvectors_in_train = False


