import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from config import NNAnalysisParameters
from dataset import get_dataloaders_for_c1c2cov
from models.Resnet import ResNet18
from models.vanilla_fc import AlphaModel
from utils import create_folder, save_model


def get_svrcov_optimizer(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)


def get_svrcov_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )


def get_svrcov_loss():
    return nn.BCEWithLogitsLoss(reduction="mean")


def get_svrcov_model(cfg: NNAnalysisParameters, device):
    img_width = 32 if cfg.dataset_type == 'CIFAR10' else 28
    input_channels = cfg.input_channels
    activation = True
    if cfg.linear_model:
        activation = False

    if cfg.model_type == 'ResNet18':
        return ResNet18(num_classes=2, input_channels=1,
                        activation=activation).to(device)
    if cfg.model_type == 'FC':
        return AlphaModel(img_width=img_width,
                          input_channels=input_channels,
                          hidden_size=cfg.hidden_size,
                          output_size=cfg.output_size,
                          num_of_layers=cfg.num_of_layers,
                          header=cfg.nn_header,
                          activation=activation).to(device)
    return None



def test_model_c1c2(model, dataloader_test_c1, dataloader_test_c2, device, loss_fn):
    model.eval()
    avg_loss = 0.0
    accuracy = 0.0
    print('Testing the model on the test set')
    with torch.no_grad():
        for x in dataloader_test_c1:
            x = x[0, ...].float().to(device)
            y = torch.tensor([1, 0])
            out = model(x)
            y = y.repeat(out.shape[0], 1).float().to(device)
            loss = loss_fn(out, y)
            avg_loss += loss.item()
            accuracy += (out.argmax(1) == y.argmax(1)).float().mean().item()

        for x in dataloader_test_c2:
            x = x[0, ...].float().to(device)
            y = torch.tensor([0, 1]).to(device)
            # y = y.repeat(x.shape[0], 1).float().to(device)
            out = model(x)
            y = y.repeat(out.shape[0], 1).float().to(device)
            loss = loss_fn(out, y)
            avg_loss += loss.item()
            accuracy += (out.argmax(1) == y.argmax(1)).float().mean().item()

    accuracy /= (len(dataloader_test_c1) + len(dataloader_test_c2))
    avg_loss /= (len(dataloader_test_c1) + len(dataloader_test_c2))
    return avg_loss, accuracy


def get_model_name(cfg: NNAnalysisParameters, c1c2):
    model_name = 'c1c2cov_' + cfg.dataset_type + '_' + cfg.model_type + \
                 '_classes=' + str(c1c2[0]) + str(c1c2[1])
    if cfg.rotate_eigenvectors_in_train:
        model_name += '_rotation'
    if cfg.same_eig:
        model_name += '_sameeig'
    if cfg.same_eigVec:
        model_name += '_sameeigVec'
    if cfg.normalize_all_db:
        model_name += '_normalldb'
    if cfg.linear_model:
        model_name += '_linear'
    if cfg.max_num_of_db_samples != np.inf:
        model_name += '_maxsamples=' + str(cfg.max_num_of_db_samples)

    model_name += '.pt'
    return model_name


def train_c1c2cov(cfg: NNAnalysisParameters, c1c2, show=False):
    create_folder('weights_c1c2/')
    create_folder('plots/')

    # Get the dataloaders
    dataloader_train_c1, \
        dataloader_test_c1, \
        dataloader_train_c2, \
        dataloader_test_c2 = get_dataloaders_for_c1c2cov(cfg, c1c2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.same_eig or cfg.same_eigVec:
        # Change the eigenvalues of C1 to be equal C2 - same with test
        eig_db2_train, eig_vec_db2_train = np.linalg.eigh(dataloader_train_c2.dataset.cov)
        eig_vec_db2_train = torch.tensor(eig_vec_db2_train.real).float()
        eig_db2_train = torch.tensor(eig_db2_train.real).float()
        eig_db1_train, eig_vec_db1_train = np.linalg.eigh(dataloader_train_c1.dataset.cov)
        eig_vec_db1_train = torch.tensor(eig_vec_db1_train.real).float()
        eig_db1_train = torch.tensor(eig_db1_train.real).float()
        if cfg.same_eigVec:
            cov1_train = eig_vec_db2_train @ torch.diag(eig_db1_train) @ eig_vec_db2_train.T
        if cfg.same_eig:
            cov1_train = eig_vec_db1_train @ torch.diag(eig_db2_train) @ eig_vec_db1_train.T
        dataloader_train_c1.dataset.cov = cov1_train

        eig_db2_test, eig_vec_db2_test = np.linalg.eigh(dataloader_test_c2.dataset.cov)
        eig_vec_db2_test = torch.tensor(eig_vec_db2_test.real).float()
        eig_db2_test = torch.tensor(eig_db2_test.real).float()
        eig_db1_test, eig_vec_db1_test = np.linalg.eigh(dataloader_test_c1.dataset.cov)
        eig_vec_db1_test = torch.tensor(eig_vec_db1_test.real).float()
        eig_db1_test = torch.tensor(eig_db1_test.real).float()
        if cfg.same_eigVec:
            cov1_test = eig_vec_db2_test @ torch.diag(eig_db1_test) @ eig_vec_db2_test.T
        if cfg.same_eig:
            cov1_test = eig_vec_db1_test @ torch.diag(eig_db2_test) @ eig_vec_db1_test.T
        dataloader_test_c1.dataset.cov = cov1_test

    # Set rotatioan mode:
    if cfg.rotate_eigenvectors_in_train:
        # Decompose cov1 to its eigenvalues
        __, eig_vec_db2 = np.linalg.eigh(dataloader_train_c2.dataset.cov)
        eig_vec_db2 = torch.tensor(eig_vec_db2.real).float()
        dataloader_train_c1.dataset.set_rotation_mode(eig_vec_db2)
        dataloader_test_c1.dataset.set_rotation_mode(eig_vec_db2)
        dataloader_test_c2.dataset.set_rotation_mode(eig_vec_db2)

    # Then define the model
    model = get_svrcov_model(cfg, device)

    # Then define the optimizer
    optimizer = get_svrcov_optimizer(cfg, model)

    # Then define the scheduler
    scheduler = get_svrcov_scheduler(cfg, optimizer)

    # Then define the loss function
    loss_fn = get_svrcov_loss()

    # Then define the training loop
    train_loss = []
    test_loss = []
    accuracy_test = []

    avg_test_loss, accuracy = \
        test_model_c1c2(model, dataloader_test_c1,
                        dataloader_test_c2, device, loss_fn)
    test_loss.append(avg_test_loss)
    accuracy_test.append(accuracy)
    model.train()

    model_name = get_model_name(cfg, c1c2)

    loop = tqdm(range(cfg.epochs), leave=True)
    for epoch in loop:
        avg_train_loss = 0.0
        counter = 0
        while counter < len(dataloader_train_c1):
            optimizer.zero_grad()

            x_c1 = next(iter(dataloader_train_c1))[0]
            x_c2 = next(iter(dataloader_train_c2))[0]
            y_c1 = torch.tensor([1, 0]).repeat(x_c1.shape[0], 1)
            y_c2 = torch.tensor([0, 1]).repeat(x_c2.shape[0], 1)
            x = torch.cat((x_c1, x_c2), 0)
            y = torch.cat((y_c1, y_c2), 0)
            # Shuffle the data
            p = torch.randperm(x.shape[0])
            x = x[p].float().to(device)
            y = y[p].float().to(device)

            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()

            optimizer.step()
            scheduler.step()
            avg_train_loss += loss.item()
            counter += 1

        avg_train_loss /= len(dataloader_train_c1)

        avg_test_loss, accuracy = \
            test_model_c1c2(model, dataloader_test_c1,
                            dataloader_test_c2, device, loss_fn)
        train_loss.append(avg_train_loss)
        test_loss.append(avg_test_loss)
        accuracy_test.append(accuracy)
        model.train()

        # update progress bar with the loss value
        loop.set_description(f"Epoch {epoch + 1}/{cfg.epochs}")
        loop.set_postfix(loss=avg_train_loss, test_loss=avg_test_loss)
        if epoch % 10 == 0:
            # Save the model
            save_model(model, path='weights_c1c2/' + model_name)

    # Then plot the results
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig('plots/c1_vs_c2_loss.png')

    if show:
        plt.show()

    # Plot accuracy
    plt.plot(accuracy_test, label='test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.savefig('plots/c1_vs_c2_accuracy.png')

    if show:
        plt.show()

    print('Accuracy: ', max(accuracy_test))
    save_model(model, path='weights_c1c2/' + model_name)

    train_loss = torch.tensor(train_loss)
    test_loss = torch.tensor(test_loss)

    return train_loss, test_loss




def run_training():
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
        for c1c2 in c1c2_list:
            train_loss, test_loss = train_c1c2cov(cfg, c1c2, show=True)


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
        for c1c2 in c1c2_list:
            train_loss, test_loss = train_c1c2cov(cfg, c1c2, show=True)


if __name__ == '__main__':
    run_training()