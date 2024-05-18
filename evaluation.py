import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import NNAnalysisParameters
from dataset import get_dataloaders_for_c1c2cov
from utils import create_folder
from train import get_svrcov_loss, get_svrcov_model, get_model_name


torch.random.manual_seed(42)


def flip_cov_mat(cov1, cov2, flip_code=(0, 0, 0, 0), eig_threshold=-600, show_norm=False):
    # First decompose the matrices
    eig1, vec1 = np.linalg.eigh(cov1)
    eig2, vec2 = np.linalg.eigh(cov2)
    # Sort in descending order
    eig1 = eig1[::-1]
    eig2 = eig2[::-1]
    vec1 = vec1[:, ::-1]
    vec2 = vec2[:, ::-1]

    # Build the new matrices according to flips_vectors
    bulk_eig_A = [eig2[:eig_threshold], eig1[:eig_threshold]][flip_code[0]]
    bulk_eig_B = [eig2[:eig_threshold], eig1[:eig_threshold]][1 - flip_code[0]]
    small_eig_A = [eig2[eig_threshold:], eig1[eig_threshold:]][flip_code[1]]
    small_eig_B = [eig2[eig_threshold:], eig1[eig_threshold:]][1 - flip_code[1]]
    bulk_vec_A = [vec2[:, :eig_threshold], vec1[:, :eig_threshold]][flip_code[2]]
    bulk_vec_B = [vec2[:, :eig_threshold], vec1[:, :eig_threshold]][1 - flip_code[2]]
    small_vec_A = [vec2[:, eig_threshold:], vec1[:, eig_threshold:]][flip_code[3]]
    small_vec_B = [vec2[:, eig_threshold:], vec1[:, eig_threshold:]][1 - flip_code[3]]

    # Build the new matrices
    A_eig = np.concatenate((bulk_eig_A, small_eig_A))
    A_vec = np.concatenate((bulk_vec_A, small_vec_A), axis=1)
    B_eig = np.concatenate((bulk_eig_B, small_eig_B))
    B_vec = np.concatenate((bulk_vec_B, small_vec_B), axis=1)

    # Build the new covariance matrices
    A = A_vec @ np.diag(A_eig) @ A_vec.T
    B = B_vec @ np.diag(B_eig) @ B_vec.T

    ## --- Check ||VV.T - I||F size --- ##
    if show_norm:
        print('||VaVa.T - I||F =', np.linalg.norm(A_vec @ A_vec.T - np.eye(A_vec.shape[0])) / A_vec.shape[0])
        print('||VbVb.T - I||F =', np.linalg.norm(B_vec @ B_vec.T - np.eye(B_vec.shape[0])) / B_vec.shape[0])
    ## -------------------------------- ##

    return A, B, A_vec, A_eig


def flip_eig_db_test_c1c2(cfg: NNAnalysisParameters,
                          c1c2=(1, 2),
                          flip_code=(0, 0, 0, 0),
                          eig_threshold=-600,
                          rotate_eigenvectors_in_test=False):
    # Define the following flips pattern: [bulk eig, small eig, bulk vec, small vec]
    # For A dataset, and B is the not code. 0 = syn, 1 = real
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model_name = get_model_name(cfg, c1c2)
    model = get_svrcov_model(cfg, device)
    if not os.path.exists('weights_c1c2/' + model_name):
        raise ValueError('Model does not exist. Train the model first!')

    model.load_state_dict(torch.load('weights_c1c2/' + model_name))

    # Define loss
    loss_fn = get_svrcov_loss()

    # Load the test data
    __, dataloader_test_c1, dataloader_train_c2, dataloader_test_c2 = \
        get_dataloaders_for_c1c2cov(cfg, c1c2)

    # Change the cov matrix of the test data according to the flip code
    cov1 = dataloader_test_c1.dataset.cov
    cov2 = dataloader_test_c2.dataset.cov

    # Set rotation mode:
    if rotate_eigenvectors_in_test:
        # Decompose cov1 and cov2 to their eigenvalues
        eig_vals1, eig_vecs1 = np.linalg.eigh(cov1)
        eig_vecs1 = eig_vecs1[:, ::-1]
        __, __, A_vec, A_eig = flip_cov_mat(cov1, cov2, flip_code, eig_threshold, show_norm=False)
        # Now, build a rotation matrix:
        # if eigenvector flips:
        if flip_code[2] == 1:
            rotation_mat = A_vec @ eig_vecs1.T
        # eigenvalues flips:
        elif flip_code[0] == 1:
            eig_ratios = A_eig / eig_vals1
            rotation_mat = eig_vecs1 @ np.diag(eig_ratios) @ eig_vecs1.T

        # Now rotate all the data of test 1
        w = dataloader_test_c1.dataset.total_db_data_samples.shape[1]
        h = dataloader_test_c1.dataset.total_db_data_samples.shape[2]

    else:
        cov1, cov2, __, __ = flip_cov_mat(cov1, cov2, flip_code, eig_threshold, show_norm=False)
        dataloader_test_c1.dataset.cov = cov1
        dataloader_test_c2.dataset.cov = cov2

    # symmetrize_cov:
    dataloader_test_c1.dataset.cov = (dataloader_test_c1.dataset.cov +
                                      dataloader_test_c1.dataset.cov.T) / 2
    dataloader_test_c2.dataset.cov = (dataloader_test_c2.dataset.cov +
                                      dataloader_test_c2.dataset.cov.T) / 2
    model.eval()

    loss_A_is_c1_avg = 0
    loss_A_is_c2_avg = 0
    accuracy_A_is_c1 = 0.0
    accuracy_A_is_c2 = 0.0
    with torch.no_grad():
        for x in dataloader_test_c1:
            x = x[0, ...].float().to(device)
            out_A = model(x)
            y_c1 = torch.tensor([1, 0]).repeat(out_A.shape[0], 1).float()
            y_c2 = torch.tensor([0, 1]).repeat(out_A.shape[0], 1).float()
            # Add to average
            loss_A_is_c1_avg += 100 * loss_fn(out_A, y_c1.to(device)).item()
            loss_A_is_c2_avg += 100 * loss_fn(out_A, y_c2.to(device)).item()
            accuracy_A_is_c1 += 100 * torch.sum(torch.argmax(out_A, dim=1) == 0).item() / out_A.shape[0]
            accuracy_A_is_c2 += 100 * torch.sum(torch.argmax(out_A, dim=1) == 1).item() / out_A.shape[0]

    # Average the results
    loss_A_is_c1_avg /= len(dataloader_test_c1)
    loss_A_is_c2_avg /= len(dataloader_test_c1)
    accuracy_A_is_c1 /= len(dataloader_test_c1)
    accuracy_A_is_c2 /= len(dataloader_test_c1)

    return (loss_A_is_c1_avg, loss_A_is_c2_avg,
            accuracy_A_is_c1, accuracy_A_is_c2)

def analyze_flip_threshold(cfg: NNAnalysisParameters,
                           c1c2_list=[(1, 2)],
                           flip_eigenvalues=False,
                           flip_eigenvectors=False,
                           rotate_eigenvectors=False,
                           holdidx=-1,
                           load_data=True):
    # Here we focus only on bulk eigenvectors flip:
    if flip_eigenvectors:
        flip_bulk_vec = 1
    else:
        flip_bulk_vec = 0
    if flip_eigenvalues:
        flip_bulk_eig = 1
    else:
        flip_bulk_eig = 0
    flip_small_eig = 0
    flip_small_vec = 0
    flip_code = (flip_bulk_eig, flip_small_eig, flip_bulk_vec, flip_small_vec)

    # We analyze different values of the threshold
    threshold_list = [0] + list(range(0, 10)) + list(np.int16(np.round(
        np.logspace(1, np.log10(cfg.img_width), 30))
    ))

    # remove duplicates
    threshold_list = list(dict.fromkeys(threshold_list))

    color_list = ['b', 'g', 'r', 'orange', 'm', 'c', 'y', 'k', 'pink', 'brown']


    iname = cfg.model_type + ('_linear' if cfg.linear_model else '') + \
            '_' + cfg.dataset_type + '_' + str(flip_eigenvectors) + \
            str(flip_eigenvalues) + ('_alldbnorm' if cfg.normalize_all_db else '') \
            + ('_maxsamples=' + str(cfg.max_num_of_db_samples) if cfg.max_num_of_db_samples != np.inf else '') \
            + ('_rotateeig' if rotate_eigenvectors else '')

    filename = 'npy/' + 'accuracy_vs_threshold' + iname + '.npy'
    if load_data and os.path.exists(filename):
        threshold_list, tot_accuracy_A_c1 = np.load(filename, allow_pickle=True)
        threshold_list = list(threshold_list)
        tot_accuracy_A_c1 = list(tot_accuracy_A_c1)

    else:

        tot_accuracy_A_c1 = []
        for i, c1c2 in enumerate(c1c2_list):
            threshold_dict = {}
            accuracy_A_c1 = []
            for eig_threshold in tqdm(threshold_list):
                AB_results = flip_eig_db_test_c1c2(
                    cfg,
                    c1c2=c1c2,
                    flip_code=flip_code,
                    eig_threshold=eig_threshold,
                    rotate_eigenvectors_in_test=rotate_eigenvectors
                )

                loss_A_is_c1_avg, loss_A_is_c2_avg, \
                    accuracy_A_is_c1, accuracy_A_is_c2 = AB_results

                threshold_dict[eig_threshold] = AB_results
                accuracy_A_c1.append(accuracy_A_is_c1)

            tot_accuracy_A_c1.append(accuracy_A_c1)

        # Save npy file of the results
        create_folder('npy/')
        np.save(filename, np.array([threshold_list, tot_accuracy_A_c1]))


    if holdidx < 0:
        plt.figure(figsize=(8, 5))
        plt.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.94)

    # Font size 12
    real_db = False
    plt.rc('font', size=13)  # controls default text sizes
    # Increase the label size and title
    plt.rc('axes', titlesize=15)  # fontsize of the axes title
    plt.rc('axes', labelsize=15)  # fontsize of the x and y labels

    for i, c1c2 in enumerate(c1c2_list):
        accuracy_A_c1 = tot_accuracy_A_c1[i]
        if str(c1c2[0]).startswith('c'):
            c1 = int(c1c2[0][1])
            c2 = int(c1c2[1][1])
            real_db = True
        else:
            c1 = int(c1c2[0])
            c2 = int(c1c2[1])

        # Save figures
        if holdidx >= 0:
            # plot with the same color:
            plt.plot(threshold_list, accuracy_A_c1,
                     label=str(cfg.max_num_of_db_samples),
                     linestyle='-', color=color_list[holdidx],
                     linewidth=2, marker='o', markersize=5)
            plt.plot(threshold_list, 100 - torch.tensor(accuracy_A_c1),
                     color=color_list[holdidx],
                     linewidth=2, marker='o', markersize=6,
                     markerfacecolor='none')
        else:
            plt.plot(threshold_list, accuracy_A_c1,
                     label=r'A = ' + str(c1) + ', ' + str(c2),
                     linestyle='-', color=color_list[i],
                     linewidth=2, marker='o', markersize=5)
            plt.plot(threshold_list, 100 - torch.tensor(accuracy_A_c1),
                     linestyle='--', color=color_list[i],
                     linewidth=2, marker='o', markersize=6,
                     markerfacecolor='none')

        plt.legend(ncol=1, loc='center right')
        plt.xlim([-0.1, np.max(threshold_list) * 10])
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(alpha=0.5)
        plt.xlabel(r'Eigenvector Threshold')
        plt.ylabel(r'Accuracy')
        plt.xscale('symlog')

    if flip_eigenvectors and flip_eigenvalues:
        flip_name = 'Flip Eigenvectors and Eigenvalues.'
    elif flip_eigenvectors:
        flip_name = 'Flip Eigenvectors'
    elif flip_eigenvalues:
        flip_name = 'Flip Eigenvalues'

    flip_name += ', ' + cfg.model_type + ', ' + cfg.dataset_type
    if holdidx > 0:
        flip_name += ', (' + str(c1c2_list[0][0]) + ', ' + str(c1c2_list[0][1]) + ')'
    if real_db:
        flip_name += ', Real DB'
    plt.title(flip_name)

    if holdidx < 0:
        create_folder('plots/')
        plt.savefig('plots/' + 'accuracy_vs_threshold' + iname + '.png')
        plt.show()


def run_eval():
    cfg = NNAnalysisParameters()
    img_size = {'FMNIST': 28, 'CIFAR10': 32}

    ### Flipping Test For Gaussians ###
    configs = [('FC', 'FMNIST', False),
               ('FC', 'CIFAR10', False),
               ('ResNet18', 'FMNIST', False),
               ('ResNet18', 'CIFAR10', False)]
    c1c2_list = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (0, 6), (0, 7), (0, 8), (0, 9)]
    for i in range(len(configs)):
        cfg.model_type, cfg.dataset_type, cfg.normalize_all_db = configs[i]
        cfg.img_width = img_size[cfg.dataset_type] ** 2
        analyze_flip_threshold(cfg, c1c2_list, flip_eigenvectors=True, flip_eigenvalues=False, load_data=True)
        analyze_flip_threshold(cfg, c1c2_list, flip_eigenvectors=False, flip_eigenvalues=True, load_data=True)


    ######## Real data test ########
    cfg = NNAnalysisParameters()
    configs = [('FC', 'FMNIST', False),
               ('FC', 'CIFAR10', False),
               ('ResNet18', 'FMNIST', False),
               ('ResNet18', 'CIFAR10', False)]
    c1c2_list = [('c0', 'c1'), ('c0', 'c2'), ('c0', 'c3'), ('c0', 'c4'), ('c0', 'c5'),
                 ('c0', 'c6'), ('c0', 'c7'), ('c0', 'c8'), ('c0', 'c9')]
    cfg.linear_model = False
    for i in range(len(configs)):
        cfg.model_type, cfg.dataset_type, cfg.normalize_all_db = configs[i]
        cfg.img_width = img_size[cfg.dataset_type] ** 2
        analyze_flip_threshold(cfg, c1c2_list, flip_eigenvectors=True, flip_eigenvalues=False, rotate_eigenvectors=True, load_data=True)
        analyze_flip_threshold(cfg, c1c2_list, flip_eigenvectors=False, flip_eigenvalues=True, rotate_eigenvectors=True, load_data=True)



if __name__ == "__main__":
    run_eval()