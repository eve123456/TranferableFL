import torch
import numpy as np
import pickle
import os
import torchvision
import argparse


class ImageDataset(object):
    def __init__(self, images, labels):
        # client data are stored as numpy.array
        # normalization will be done at worker_utils.MiniDataset
        if isinstance(images, torch.Tensor):
            self.data = images.numpy()
        else:
            self.data = images
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
            data_lst.append(data[i: i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i: i+delta])
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


def main(args):
    NUM_USER = args.num_client
    output_path = args.output
    dataset_name = args.dataset
    
    print('>>> Downloading data.')
    if dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=output_path, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=output_path, train=False, download=True)
    elif dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=output_path, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=output_path, train=False, download=True)
    # elif dataset_name == 'cifar100':
    #     trainset = torchvision.datasets.CIFAR100(root=output_path, train=True, download=True)
    #     testset = torchvision.datasets.CIFAR100(root=output_path, train=False, download=True)
    elif dataset_name == 'svhn':
        trainset = torchvision.datasets.SVHN(root=output_path, split='train', download=True)
        testset = torchvision.datasets.SVHN(root=output_path, split='test', download=True)
    else:
        raise NotImplementedError

    train_mnist = ImageDataset(trainset.data, trainset.targets)
    test_mnist = ImageDataset(testset.data, testset.targets)

    mnist_traindata = []
    for number in range(10):
        idx = train_mnist.target == number
        mnist_traindata.append(train_mnist.data[idx])
    
    mnist_testdata = []
    for number in range(10):
        idx = test_mnist.target == number
        mnist_testdata.append(test_mnist.data[idx])
    
    # save the aggregated dataset
    data_X_all = np.concatenate(mnist_traindata + mnist_testdata).tolist()
    data_y_all = []
    for label_idx in range(10):
        data_y_all += [label_idx] * mnist_traindata[label_idx].shape[0]
    for label_idx in range(10):
        data_y_all += [label_idx] * mnist_testdata[label_idx].shape[0]

    data_all = {'users': [], 'user_data': {}, 'num_samples': []}
    data_all['users'].append(0)
    data_all['user_data'][0] = {'x': data_X_all, 'y': data_y_all}
    data_all['num_samples'].append(len(data_y_all))
    
    data_path = os.path.join(output_path, dataset_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    data_all_path = os.path.join(data_path, 'all_data.pkl')
    with open(data_all_path, 'wb') as outfile:
        pickle.dump(data_all, outfile)

    print('>>> Save aggregated data.')
    
    min_number = min([len(dig) for dig in mnist_traindata])
    for number in range(10):
        mnist_traindata[number] = mnist_traindata[number][:min_number]

    split_mnist_traindata = []
    for digit in mnist_traindata:
        split_mnist_traindata.append(data_split(digit, 20))

    split_mnist_testdata = []
    for digit in mnist_testdata:
        split_mnist_testdata.append(data_split(digit, 20))
    
    data_distribution = np.array([len(v) for v in mnist_traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_mnist_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_mnist_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(NUM_USER):
        print(user, np.array([len(v) for v in split_mnist_traindata]))

        for d in choose_two_digit(split_mnist_traindata):
            l = len(split_mnist_traindata[d][-1])
            train_X[user] += split_mnist_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_mnist_testdata[d][-1])
            test_X[user] += split_mnist_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    # Setup directory for train/test data
    print('>>> Set data path.')
    train_path = os.path.join(data_path, 'train', 'all_data_equal_niid.pkl')
    test_path = os.path.join(data_path, 'test', 'all_data_equal_niid.pkl')
    
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
    with open(train_path, 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_data, outfile)

    print('>>> Save data.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset', type=str)
    parser.add_argument('--output', help='path of output', type=str, default='data')
    parser.add_argument('--num_client', help='number of clients', type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)
