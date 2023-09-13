import os
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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


def get_loader(image_dir, dataset_name, batch_size, **kwargs):
    assert dataset_name == 'mnist-m'

    train_list = os.path.join(image_dir, 'mnist_m_train_labels.txt')
    test_list = os.path.join(image_dir, 'mnist_m_test_labels.txt')

    img_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        lambda x: x * 255,
        transforms.Grayscale()
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

    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs)

    return train_loader, test_loader
