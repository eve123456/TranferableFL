import numpy as np
import argparse
import importlib
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from src.utils.worker_utils import read_data
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS

from data.data_loader import GetLoader

import warnings

warnings.filterwarnings('ignore')

def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg4')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_1_random_niid')
                        # default='svhn_all_data_1_random_niid')
    parser.add_argument('--finetune_dataset',
                        help='name of finetune dataset;',
                        type=str,
                        # default='mnist-m')
                        default = 'mnist')
    parser.add_argument('--finetune_lr',
                        help='lr for finetune;',
                        type=float,
                        default=0.01)
    parser.add_argument('--finetune_wd',
                        help='weight decay for finetune;',
                        type=float,
                        default=0.0)
    parser.add_argument('--finetune_epochs',
                        help='epochs for finetune;',
                        type=int,
                        # default=100)
                        default = 2)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='LeNet')
                        # default = 'cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=True,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        # default=1000)
                        default = 2)
                        
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
                        
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
                       
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
                        # default =1)
    
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


# Parse command line arguments
options, trainer_class, dataset_name, sub_data = read_options()
options["dataset_name"] = dataset_name

train_path = os.path.join('./data', dataset_name, 'data', 'train')
test_path = os.path.join('./data', dataset_name, 'data', 'test')

# `dataset` is a tuple like (cids, groups, train_data, test_data)
all_data_info = read_data(train_path, test_path, dataset_name, sub_data)

# Call appropriate trainer
trainer = trainer_class(options, all_data_info)
trainer.train()

# FL training finish here, save the latest server model
flat_model_param = trainer.latest_model
PATH = f"./models/{options['model']}_{dataset_name}_{options['algo']}"
torch.save(flat_model_param, PATH)



from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
from src.models.model import choose_model
from ft_functions import *

# This is the final flattened model params
flat_model_params = trainer.latest_model

# Optional: load from the saved model.
PATH = f"./models/{options['model']}_{dataset_name}_{options['algo']}"
flat_model_params = torch.load(PATH)


# Initialize new models
model_source_only = choose_model(options) # baseline: lower bound
model_ft = choose_model(options) # baseline: standard fine-tune
model_target_only = choose_model(options) # baseline: upper bound
model_flag = choose_model(options) # test if the model archetecture is proper for target domain
# If performances of model_flag and model_target_only deviates, then the model is proper for target domain.

# Assign model params
set_flat_params_to(model_source_only, flat_model_params)
set_flat_params_to(model_ft, flat_model_params)
# Now model is set with flat_model_params

# Start fine-tuning below
# First, freeze all but last layer
def freeze(model):
    for mod in model.children():
        for params in mod.parameters():
            params.requires_grad = False

    for params in mod.parameters():
        params.requires_grad = True

freeze(model_flag)
freeze(model_ft)


# # Next fine-tune the model on mnist:
device = torch.device(f"cuda:{options['device']}")
criterion = torch.nn.CrossEntropyLoss()
kwargs = {'num_workers': 16}




# Prepare data
# assert options['finetune_dataset'] == 'mnist-m' # now we only support transfer from svhn to mnist.
if options['finetune_dataset'] == "mnist-m":
    target_image_root = './data/TARGET/mnist-m/mnist_m'

    tg_train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    tg_test_list = os.path.join(target_image_root, 'mnist_m_test_labels.txt')
    
    img_transform_target = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    lambda x: x*255,
    transforms.Grayscale()
    ])

    tg_trainset = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=tg_train_list,
        transform=img_transform_target
    )

    tg_testset = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_test'),
        data_list=tg_test_list,
        transform=img_transform_target
    )

    tg_train_loader = torch.utils.data.DataLoader(
        dataset=tg_trainset,
        batch_size=options['batch_size'],
        shuffle=True,
        **kwargs)

    tg_test_loader = torch.utils.data.DataLoader(
        dataset=tg_testset,
        batch_size=options['batch_size'],
        shuffle=True,
        **kwargs)

elif options['finetune_dataset'] == "mnist":
    data_path = './data/TARGET/mnist'
    
    img_transform_target = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x*255,
    ])
    
    trainset = torchvision.datasets.MNIST(root=data_path, train =True, download=False, transform=img_transform_target)
    testset = torchvision.datasets.MNIST(root=data_path, train =False, download=False, transform=img_transform_target)
    tg_train_loader = torch.utils.data.DataLoader(trainset, batch_size=options['batch_size'], shuffle=True, **kwargs)
    tg_test_loader = torch.utils.data.DataLoader(testset, batch_size=options['batch_size'], shuffle=True, **kwargs)



####### Train
### check if feature extractor of the model matters for target fomain

# # Train model_target_only
# print(f"Training model_target_only...")
# ft_trn_main(model_target_only, options, device,tg_train_loader, tg_test_loader, criterion)

# # Train model_flag
# print(f"Training model_flag...")
# ft_trn_main(model_flag, options, device,tg_train_loader, tg_test_loader, criterion)

### Train model_ft
print(f"Training model_ft...")
ft_trn_main(model_ft, options, device,tg_train_loader, tg_test_loader, criterion)
### Evaluate model_source_only
res_sc_only = torch.zeros((2))
print(f"Evaluating model_source_only...")
model_source_only = model_source_only.to(device)
res_sc_only[0], res_sc_only[1] = ft_eval(model_source_only,device,tg_test_loader, criterion)
print(f"Tst_loss:{res_sc_only[0].item():.4f}, Tst_acc:{res_sc_only[1].item():.4f}")

# Now save the final transfered model
PATH = f"./models/{options['model']}_{dataset_name}_{options['algo']}_finetune_{options['finetune_dataset']}"
torch.save(model_ft.state_dict(), PATH)
