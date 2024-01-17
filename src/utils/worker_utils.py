import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image


__all__ = ['mkdir', 'read_data', 'Metrics', "MiniDataset"]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def read_data(train_data_dir, test_data_dir=None, key=None, dataset_name='mnist', DA = None, DA_type = None):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    if key is not None:
        train_files = list(filter(lambda x: str(key) in x, train_files))

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        train_data[cid] = MiniDataset(v['x'], v['y'], dataset_name, DA = DA, DA_type = DA_type)
        # transformation assigned, but no transformation implemented yet

    if test_data_dir is not None:
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.pkl')]
        if key is not None:
            test_files = list(filter(lambda x: str(key) in x, test_files))

        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            print('    ', file_path)

            with open(file_path, 'rb') as inf:
                cdata = pickle.load(inf)
            test_data.update(cdata['user_data'])

        for cid, v in test_data.items():
            test_data[cid] = MiniDataset(v['x'], v['y'], dataset_name, DA = None)
    else:
        test_data = None

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class MiniDataset(Dataset):
    def __init__(self, data, labels, dataset_name, DA = None, DA_type = None):
        super(MiniDataset, self).__init__()
        self.data = np.array(data).astype("uint8")  # images
        self.labels = np.array(labels).astype("int64")
        
        if dataset_name == "cifar10":
            # 32 * 32
            if DA:
                if DA_type == "HF":
                    self.transform = self.transform = transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                        ]
                    )
                
                if DA_type == "VF":
                    self.transform = transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                        ]
                    )
                
                if DA_type == "R":
                    self.transform = transforms.Compose(
                        [
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                        ]
                    )
                
                if DA_type == "CJ":
                    self.transform = transforms.Compose(
                        [
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                        ]
                    )
                
                if DA_type == "Crop":
                    self.transform = transforms.Compose(
                        [
                            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                        ]
                    )

                # self.transform = transforms.Compose(
                #     [
                #         # add data augmentation
                #         transforms.RandomHorizontalFlip(),
                #         # transforms.RandomVerticalFlip(),
                #         # transforms.RandomRotation(15),
                #         # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                #         # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                #         # transforms.ToTensor(),
                #         # transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                #     ]
                # )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                    ]
                )



        elif dataset_name == "cifar100":
            # 32 * 32
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.50707516, 0.48654887, 0.44091784], [0.26733429, 0.25643846, 0.27615047])
                ]
            )
            
        elif dataset_name == "svhn":
            # 32 * 32
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
                ]
            )
            
        elif dataset_name == "mnist":
            # 28 * 28
            if self.data.ndim == 4 and self.data.shape[3] == 1:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            elif self.data.ndim == 3:
                self.data = self.data.reshape((-1, 28, 28, 1)).astype("uint8")
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            else:
                raise NotImplementedError
                
        else:
            
            raise NotImplementedError
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class Metrics(object):
    def __init__(self, clients, options, name=''):
        self.options = options
        num_rounds = options['num_round'] + 1
        self.bytes_written = {c.cid: [0] * num_rounds for c in clients}
        self.client_computations = {c.cid: [0] * num_rounds for c in clients}
        self.bytes_read = {c.cid: [0] * num_rounds for c in clients}

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.gradnorm_on_train_data = [0] * num_rounds
        self.graddiff_on_train_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset']))
        suffix = '{}_sd{}_lr{}_ep{}_bs{}_{}'.format(name,
                                                    options['seed'],
                                                    options['lr'],
                                                    options['num_epoch'],
                                                    options['batch_size'],
                                                    'w' if options['noaverage'] else 'a')

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        if options['dis']:
            suffix = options['dis']
            self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        # self.gradnorm_on_train_data[round_i] = train_stats['gradnorm']
        # self.graddiff_on_train_data[round_i] = train_stats['graddiff']

        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)
        # self.train_writer.add_scalar('gradnorm', train_stats['gradnorm'], round_i)
        # self.train_writer.add_scalar('graddiff', train_stats['graddiff'], round_i)

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.options['dataset']
        metrics['num_round'] = self.options['num_round']
        metrics['eval_every'] = self.options['eval_every']
        metrics['lr'] = self.options['lr']
        metrics['num_epoch'] = self.options['num_epoch']
        metrics['batch_size'] = self.options['batch_size']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['gradnorm_on_train_data'] = self.gradnorm_on_train_data
        metrics['graddiff_on_train_data'] = self.graddiff_on_train_data

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read

        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(str(metrics), ouf)
