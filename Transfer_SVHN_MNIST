import numpy as np
import argparse
import importlib
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler

from src.utils.worker_utils import read_data
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


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
                        default='svhn_all_data_1_random_niid')
    parser.add_argument('--finetune_dataset',
                        help='name of finetune dataset;',
                        type=str,
                        default='mnist')
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
                        default='cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
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
                        # default=10)
                        default = 1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        # default=10)
                        default = 1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
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
    parsed = parser.parse_args([])
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

# This is the final flattened model params
flat_model_params = trainer.latest_model

# Optional: load from the saved model.
# PATH = f"./models/{options['model']}_{dataset_name}_{options['algo']}"
# flat_model_params = torch.load(PATH)


# Initialize a new model
model = choose_model(options)
# Assign model params
set_flat_params_to(model, flat_model_params)
# Now model is set with flat_model_params

# Start fine-tuning below
# First, freeze all but last layer
for mod in model.children():
    for params in mod.parameters():
        params.requires_grad = False

for params in mod.parameters():
    params.requires_grad = True


import torchvision
import torchvision.transforms as transforms


# # Next fine-tune the model on mnist:

device = torch.device(f"cuda:{options['device']}")
criterion = torch.nn.CrossEntropyLoss()

# Prepare data
assert options['finetune_dataset'] == 'mnist' # now we only support transfer from svhn to mnist.
data_path = './data/TARGET/mnist'
kwargs = {'num_workers': 16}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2),
    transforms.Normalize((0.1307,), (0.3081,)) ,
])


trainset = torchvision.datasets.MNIST(root=data_path, train =True, download=True, transform=transform)

testset = torchvision.datasets.MNIST(root=data_path, train =False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=options['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=options['batch_size'], shuffle=True, **kwargs)

# Prepare model: just put it to device
model = model.to(device)

# Prepare optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=options['finetune_lr'], weight_decay=options['finetune_wd']) 

def ft_train(model,optimizer,device,train_loader):
    model.train()
    for x,y in train_loader:
        optimizer.zero_grad()
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        del pred

def ft_eval(model,optimizer,device,data_loader):
    model.eval()
    acc, loss, total = 0, 0, 0
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(y).sum().item()
        target_size = y.size(0)
        loss += loss.item() * y.size(0)
        acc += correct
        total += target_size

        del pred
        
    total_loss = loss/total
    total_acc = acc/total
    
    return total_loss, total_acc

# training loops
results = torch.zeros((options['finetune_epochs'],4)) # train_loss, train_acc, test_loss, test_acc
for epoch in range(options['finetune_epochs']):
    # Train 1 epoch
    ft_train(model,optimizer,device,train_loader)
    
    # Get train stats
    results[epoch,0], results[epoch,1] = ft_eval(model,optimizer,device,train_loader)
    
    # Get test stats
    results[epoch,2], results[epoch,3] = ft_eval(model,optimizer,device,test_loader)
    
    print(f"Epoch:{epoch+1:03d}, Trn_loss:{results[epoch,0].item():.4f}, Trn_acc:{results[epoch,1].item():.4f}, Tst_loss:{results[epoch,2].item():.4f}, Tst_acc:{results[epoch,3].item():.4f}")

print("Finetune done")
# Now save the final model
PATH = f"./models/{options['model']}_{dataset_name}_{options['algo']}_finetune_{options['finetune_dataset']}"
torch.save(model.state_dict(), PATH)
