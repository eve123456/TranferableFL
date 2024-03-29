import torch
import numpy as np
import pickle
import os
import torchvision
import torchvision.transforms as transforms
cpath = os.path.dirname(__file__)


NUM_USER = 10
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = not False
DOWNLOAD = False
np.random.seed(6)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst


def main():
    # Get SVHN data, normalize, and divide by level
    print('>>> Get SVHN data.')
    

    transform = transforms.Compose([
    transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ])

    trainset = torchvision.datasets.SVHN(root=DATASET_FILE, split='train', download=DOWNLOAD, transform=transform)
    testset = torchvision.datasets.SVHN(root=DATASET_FILE, split='test', download=DOWNLOAD, transform=transform)

    train_svhn = ImageDataset(trainset.data, trainset.labels)
    test_svhn = ImageDataset(testset.data, testset.labels)

    svhn_traindata = []
    for number in range(10):
        idx = train_svhn.target == number
        svhn_traindata.append(train_svhn.data[idx])
    split_svhn_traindata = []
    for digit in svhn_traindata:
        split_svhn_traindata.append(data_split(digit, int(NUM_USER*2/10)))

    svhn_testdata = []
    for number in range(10):
        idx = test_svhn.target == number
        svhn_testdata.append(test_svhn.data[idx])
    split_svhn_testdata = []
    for digit in svhn_testdata:
        split_svhn_testdata.append(data_split(digit, int(NUM_USER*2/10)))

    data_distribution = np.array([len(v) for v in svhn_traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_svhn_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_svhn_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(NUM_USER):
        print(f"Client {user} choosing from distribution {np.array([len(v) for v in split_svhn_traindata])}")
        dl = []
        for d in choose_two_digit(split_svhn_traindata):
            dl.append(d)
            l = len(split_svhn_traindata[d][-1])
            train_X[user] += split_svhn_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_svhn_testdata[d][-1])
            test_X[user] += split_svhn_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()
        
        print(f"Client {user} got digits: {dl}")

    # Setup directory for train/test data
    print('>>> Set data path for SVHN.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/data/train/all_data_{}_random_niid.pkl'.format(cpath, image)
    test_path = '{}/data/test/all_data_{}_random_niid.pkl'.format(cpath, image)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 1000 users
    for i in range(NUM_USER):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')


if __name__ == '__main__':
    main()
