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


    

def get_loader_mnist_m(image_dir, dataset_name, batch_size, **kwargs):
    assert dataset_name == 'mnist-m'

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

    return train_loader, test_loader




def get_loader_svhn(dataset_name, batch_size, **kwargs):
    assert dataset_name == 'svhn'

    print('>>> Get SVHN data.')
    

    # # Next fine-tune the model on svhn:

    data_path = './data/SVHN'
    kwargs = {'num_workers': 16}
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ])
    trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=False, transform=transform)
    # extraset = torchvision.datasets.SVHN(root=data_path, split='extra', download=True, transform=transform)
    # trainset = torch.utils.data.ConcatDataset([trainset, extraset])
    testset = torchvision.datasets.SVHN(root=data_path, split='test', download=False, transform=transform)



    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)


    return train_loader, test_loader



def get_loader_mnist(dataset_name, batch_size, **kwargs):
    assert dataset_name == 'mnist'

    print('>>> Get MNIST data.')
    

    # # Next fine-tune the model on svhn:

    data_path = './data/MNIST'
    kwargs = {'num_workers': 16}
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ])
    trainset = torchvision.datasets.MNIST(root=data_path, split='train', download=False, transform=transform)
    # extraset = torchvision.datasets.SVHN(root=data_path, split='extra', download=True, transform=transform)
    # trainset = torch.utils.data.ConcatDataset([trainset, extraset])
    testset = torchvision.datasets.MNIST(root=data_path, split='test', download=False, transform=transform)



    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)


    return train_loader, test_loader