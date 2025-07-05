import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets

from config import NNAnalysisParameters

def get_dataloaders_for_c1c2cov(cfg: NNAnalysisParameters, c1c2):
    c1, c2 = c1c2
    dg_train_c1 = DataSampler(class_number=c1,
                              num_batch_samples=cfg.num_batch_samples,
                              batch_size=cfg.batch_size,
                              max_num_of_db_samples=cfg.max_num_of_db_samples,
                              normalize_all_db=cfg.normalize_all_db,
                              train=True,
                              gray=True,
                              db_name=cfg.dataset_type)
    dg_train_c2 = DataSampler(class_number=c2,
                              num_batch_samples=cfg.num_batch_samples,
                              batch_size=cfg.batch_size,
                              normalize_all_db=cfg.normalize_all_db,
                              max_num_of_db_samples=cfg.max_num_of_db_samples,
                              train=True,
                              gray=True,
                              db_name=cfg.dataset_type)
    dg_test_c1 = DataSampler(class_number=c1,
                             num_batch_samples=cfg.num_batch_samples,
                             batch_size=cfg.batch_size,
                             normalize_all_db=cfg.normalize_all_db,
                             max_num_of_db_samples=cfg.max_num_of_db_samples,
                             train=False,
                             gray=True,
                             db_name=cfg.dataset_type)
    dg_test_c2 = DataSampler(class_number=c2,
                             num_batch_samples=cfg.num_batch_samples,
                             batch_size=cfg.batch_size,
                             normalize_all_db=cfg.normalize_all_db,
                             max_num_of_db_samples=cfg.max_num_of_db_samples,
                             train=False,
                             gray=True,
                             db_name=cfg.dataset_type)

    dataloader_train_c1 = torch.utils.data.DataLoader(
        dg_train_c1, batch_size=1,
        shuffle=False, pin_memory=True, drop_last=False)
    dataloader_test_c1 = torch.utils.data.DataLoader(
        dg_test_c1, batch_size=1,
        shuffle=False, pin_memory=True, drop_last=False)
    dataloader_train_c2 = torch.utils.data.DataLoader(
        dg_train_c2, batch_size=1,
        shuffle=False, pin_memory=True, drop_last=False)
    dataloader_test_c2 = torch.utils.data.DataLoader(
        dg_test_c2, batch_size=1,
        shuffle=False, pin_memory=True, drop_last=False)

    return (dataloader_train_c1, dataloader_test_c1,
            dataloader_train_c2, dataloader_test_c2)


# def get_random_data(train):
#     # This function takes
#     db = datasets.CIFAR10(root='../datasets',
#                                train=train,
#                                download=True,
#                                transform=None)
#     # Now, run over all the


class DataSampler(Dataset):
    def __init__(self, class_number=0, num_batch_samples=5, batch_size=1,
                 max_num_of_db_samples=np.inf, gray=True, normalization=True,
                 normalize_all_db=False, train=True, db_name='CIFAR10'):
        super().__init__()
        self.gray = gray
        self.num_batch_samples = num_batch_samples
        self.db_name = db_name

        self.original_data_mode = False
        if str(class_number)[0].lower() == 'c':
            class_number = int(str(class_number)[1:])
            self.original_data_mode = True

        self.mixed_lower_moment = False
        if (str(class_number)[0].lower() == 'm' or
                str(class_number)[0].lower() == 'n'):
            class_number = int(str(class_number)[1:])
            second_class = (class_number +
                            (1 if str(class_number)[0].lower() == 'm' else -1)
                            ) % 10
            self.mixed_lower_moment = True

        self.class_number = class_number
        self.batch_size = batch_size
        self.max_num_of_db_samples = max_num_of_db_samples
        self.img_size = (32 if self.db_name == 'CIFAR10' else 28)
        self.normalization = normalization
        self.normalize_all_db = normalize_all_db
        self.train = train
        if self.class_number == -1:
            self.load_random_db()
        else:
            self.load_db()
        self.set_cov_matrix()
        self.rotate_eigenvectors = False

        if self.mixed_lower_moment:
            self.datasampler2 = DataSampler(
                class_number=second_class,
                num_batch_samples=num_batch_samples,
                batch_size=batch_size,
                max_num_of_db_samples=max_num_of_db_samples,
                gray=gray,
                normalization=normalization,
                normalize_all_db=normalize_all_db,
                train=train,
                db_name = db_name
            )


    def load_random_db(self):
        # Generate completely random data
        data_per_class = 5000
        data_array = np.random.randn(data_per_class, 1, self.img_size, self.img_size)
        # Add another normalization to normal distribution (0, 1)
        data_array -= data_array.mean()
        data_array /= data_array.std()
        self.total_db_data_samples = data_array

        # labels = -1
        labels_array = np.ones(data_per_class) * self.class_number
        self.total_db_labels = torch.from_numpy(labels_array).long()

    def load_db(self):
        if self.db_name == 'CIFAR10':
            db = datasets.CIFAR10(root='../datasets',
                                       train=self.train,
                                       download=True,
                                       transform=None)
        elif self.db_name == 'FMNIST':
            db = datasets.FashionMNIST(root='../datasets',
                                       train=self.train,
                                       download=True,
                                       transform=None)

        # elif self.db_name == 'NOISE':
        #     db = get_random_data()

        # Take only the data of the class we want
        idxs = np.where(np.array(db.targets) == self.class_number)
        db.data = db.data[idxs]
        db.targets = np.array(db.targets)[idxs]

        if self.max_num_of_db_samples < len(db.data):
            # Take only the first max_num_of_db_samples
            db.data = db.data[:self.max_num_of_db_samples]
            db.targets = db.targets[:self.max_num_of_db_samples]

        data_array = np.array(db.data)
        labels_array = np.array(db.targets)
        # Shuffle data and the target together
        p = np.random.permutation(len(data_array))
        data_array = data_array[p]
        labels_array = labels_array[p]

        # Check for normalization with Gaussian distribution
        data_array = (data_array / 255.).astype(np.float32) * 2 - 1
        # Convert to grayscale
        if self.gray and self.db_name == 'CIFAR10':
            data_array = np.dot(data_array[..., :3], [0.299, 0.587, 0.114])

        if self.normalize_all_db:
            alldb = (db.data / 255.) * 2 - 1
            if 'torch.float32' != str(alldb.dtype):
                alldb = torch.tensor(alldb).float()
            if self.gray and self.db_name == 'CIFAR10':
                alldb = alldb @ torch.tensor([0.299, 0.587, 0.114])

            mean_db = torch.mean(alldb).numpy()
            std_db = torch.std(alldb).numpy()
            data_array = (data_array - mean_db) / std_db
        elif self.normalization:
            # Add another normalization to normal distribution (0, 1)
            data_array -= data_array.mean()
            data_array /= data_array.std()

        # move channels from (B, H, W ,C) to (B, C, H, W)
        data_array = np.moveaxis(data_array, -1, 1)

        self.total_db_data_samples = data_array
        self.total_db_data_targets = labels_array

    def set_cov_matrix(self):
        # From the entire db, calculate the covariance matrix for each class
        dvec = self.total_db_data_samples.reshape(self.total_db_data_samples.shape[0], -1)

        # Calculate the covariance matrix of each class
        self.cov = np.cov(dvec, rowvar=False)

        # Make PSD
        eigval, eigvec = np.linalg.eigh(self.cov)
        eigval = np.abs(eigval)
        self.cov = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
        self.cov = (self.cov + self.cov.T) / 2  # make it symmetric Just for numeric error!

        # TODO - for the review (22.2.25) --- Just test for the new random eigvectors ---
        if False:
            # Decompose the covariance matrix to its eigenvectors and eigenvalues
            eigval, eigvec = np.linalg.eigh(self.cov)
            # Load matrices
            new_eigvec = np.load('random_matrices.npy')[self.class_number]
            # Compose again the covariance matrix
            self.cov = new_eigvec.dot(np.diag(eigval)).dot(new_eigvec.T)

        # check if the covariance matrix is positive definite
        if not np.isclose(self.cov, self.cov.T).all():
            print('covariance matrix is not positive definite:', self.cov)

    def __len__(self):
        if self.original_data_mode or self.mixed_lower_moment:
            return len(self.total_db_data_samples) // self.batch_size
        return self.num_batch_samples

    def sample_cov(self):
        # Sample from the covariance matrix of each class
        self.mean = np.zeros(self.cov.shape[0])
        try:
            sample = np.random.multivariate_normal(self.mean, self.cov, self.batch_size)
            sample = sample.reshape(self.batch_size, 1, self.img_size, self.img_size)
        except:
            print('covariance matrix is not positive definite:', self.cov)
        # sample = np.random.multivariate_normal(self.mean, self.cov, self.batch_size)
        if self.rotate_eigenvectors:
            sample = self.rotate_samples(sample)
        return sample

    def __getitem__(self, idx):
        if self.original_data_mode:
            # take batch of self.size_batch from the data
            return self.total_db_data_samples[idx * self.batch_size:(idx + 1) * self.batch_size, np.newaxis]
        if self.mixed_lower_moment:
            # Calculate the image - sample from the cov2 + sample from cov1
            S1 = self.sample_cov()
            S2 = self.datasampler2.sample_cov()
            I = self.total_db_data_samples[idx * self.batch_size:(idx + 1) * self.batch_size, np.newaxis]
            return I - S1 + S2
        return self.sample_cov()



