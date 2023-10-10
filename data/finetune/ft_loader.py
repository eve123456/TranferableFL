import os
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import scipy


class FTDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


def get_loader(image_dir, dataset_name, batch_size, extra_train=False, **kwargs):
    
    if dataset_name == 'mnist-m':
        
        train_list = os.path.join(image_dir, 'mnist_m_train_labels.txt')
        test_list = os.path.join(image_dir, 'mnist_m_test_labels.txt')

        img_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            # lambda x: x * 255,
            transforms.Grayscale(),
            transforms.Normalize((0.4549,), (0.2208,))  # These are the mean and std computed on mnist_m_train
        ])

        trainset = FTDataset(
            data_root=os.path.join(image_dir, 'mnist_m_train'),
            data_list=train_list,
            transform=img_transform
        )

        testset = FTDataset(
            data_root=os.path.join(image_dir, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform
        )

        train_loader = DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs)

        test_loader = DataLoader(
            dataset=testset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs)
    
    elif dataset_name == 'mnist':
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root=image_dir, train=True, download=True, transform=img_transform)
        testset = torchvision.datasets.MNIST(root=image_dir, train=False, download=True, transform=img_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    
    elif dataset_name == 'svhn':
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])
        trainset = torchvision.datasets.SVHN(root=image_dir, split='train', download=True, transform=img_transform)
        if extra_train:
            extraset = torchvision.datasets.SVHN(root=image_dir, split='extra', download=True, transform=img_transform)
            trainset = torch.utils.data.ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=image_dir, split='test', download=True, transform=img_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    
    elif dataset_name == 'cifar10':
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
        ])
        trainset = torchvision.datasets.CIFAR10(root=image_dir, train=True, download=True, transform=img_transform)
        testset = torchvision.datasets.CIFAR10(root=image_dir, train=False, download=True, transform=img_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    
    elif dataset_name == 'cifar100':
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.50707516, 0.48654887, 0.44091784], [0.26733429, 0.25643846, 0.27615047])
        ])
        trainset = torchvision.datasets.CIFAR100(root=image_dir, train=True, download=True, transform=img_transform)
        testset = torchvision.datasets.CIFAR100(root=image_dir, train=False, download=True, transform=img_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    
    else:
        
        raise NotImplementedError
    
    return train_loader, test_loader
