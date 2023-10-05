import numpy as np
import pickle
import json
import datetime
import argparse
import warnings
import importlib
import torch
import time
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian
from torch.nn.utils import parameters_to_vector
from data.finetune.ft_loader import get_loader_mnist_m
from data.finetune.ft_loader import get_loader_svhn
from src.utils.worker_utils import read_data
from src.models.model import choose_model
from src.trainers.finetune import ft_train, eval
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
from pyhessian import hessian

warnings.filterwarnings('ignore')


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavgtl')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_1_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='lenet')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.0)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
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
                        help='whether to only average local solutions (default: False)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default='0',
                        type=str)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=100)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    
    parser.add_argument('--repeat_epoch',
                        help = 'scale up num_epoch in local worker, ~10 for 100 clients with batch size 64, ~1 for clients with batch size > local datasize',
                        type = int,
                        default = 10)
    
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    
    parser.add_argument('--opt_lr',
                        help='flag for optimizing local learning rate at each round (default: False);',
                        action='store_true',
                        default=False)
    
    parser.add_argument('--reg_max',
                        help='flag for regularizing J with max-norm local J.',
                        action='store_true',
                        default=False)
    
    parser.add_argument('--reg_J',
                        help='flag for regularizing Jacobian in the form of upper bound (default: False).',
                        action='store_true',
                        default=False)
    parser.add_argument('--reg_J_coef',
                        help='coefficient for regularization on Jacobian;',
                        type=float,
                        default=0.0)
    
    parser.add_argument('--reg_J_norm_coef',
                        help='coefficient for regularization on Jacobian norm;',
                        type=float,
                        default=0.0)
    
    parser.add_argument('--reg_J_ind_coef',
                        help='coefficient for regularization on Jacobian (indexwise);',
                        type=float,
                        default=0.01)

    parser.add_argument('--clip',
                        help = 'flag for whether do grad norm clip',
                        action = 'store_true',
                        default = False)
   
    parser.add_argument('--ft_dataset',
                        help='dataset for fine-tuning;',
                        type=str,
                        default='mnist-m')
    parser.add_argument('--ft_epochs',
                        help='epochs for fine-tuning;',
                        type=int,
                        default=200)
    parser.add_argument('--ft_batch_size',
                        help='batch size for fine-tuning;',
                        type=int,
                        default=128)
    parser.add_argument('--ft_lr',
                        help='learning rate for fine-tuning;',
                        type=float,
                        default=1e-3)
    parser.add_argument('--ft_wd',
                        help='weight decay for fine-tuning;',
                        type=float,
                        default=1e-2)
    parser.add_argument('--n_init',
                        help='number of initial models to consider;',
                        type=int,
                        default=10)
    parser.add_argument('--alpha',
                        help='estimate of lipschitz continuous gradient constant (0 means no estimate yet);',
                        type=float,
                        default=13.2316427)
    parser.add_argument('--last_k',
                        help='number of fc layers to fine-tune;',
                        type=int,
                        default=1)
    parser.add_argument('--early_stopping',
                        help='number of epochs for early stopping during training (0 means no early stopping);',
                        type=int,
                        default=10)
    parser.add_argument('--noft',
                        action='store_true',
                        default=False,
                        help='not performing fine-tuning (default: False);')
    parser.add_argument('--repeat',
                        help='number of repeated trails;',
                        type=int,
                        default=5)
    


    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    if not options['gpu']:
        options['device'] = 'cpu'
    else:
        options['device'] = 'cuda:' + options['device']
    print('Using device: ' + options['device'])

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


def eval_hessian(model, data_loader, criterion, device, compute_alpha):
    # only compute alpha (spectral norm of hessian) when the estimate is unknown
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_num = 0
    alpha_list = []
    for inputs, targets in data_loader:
        # we use cuda to make the computation fast
        inputs, targets = inputs.to(device), targets.to(device)
        pred = model(inputs)
        loss = criterion(pred, targets)
        total_loss += loss.item() * targets.size(0)
        total_num += targets.size(0)
        
        if compute_alpha:
            # create the hessian computation module
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            alpha_list.append(top_eigenvalues[-1])
    
    total_loss /= total_num
    if len(alpha_list) == 0:
        alpha = 0.0
    else:
        alpha = max(alpha_list)
    return total_loss, alpha


def freeze(model, k):
    # only fine-tune the last k fc layers (if there are more than k fc layers)
    num_layer = 0
    for mod in model.children():
        for params in mod.parameters():
            params.requires_grad = False
        num_layer += 1

    for mod in model.children():
        num_layer -= 1
        if num_layer <= k and isinstance(mod, torch.nn.Linear):
            for params in mod.parameters():
                params.requires_grad = True


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()

    # first estimate the value for alpha
    data_all_path = os.path.join('./data', dataset_name, 'data')
    _, _, data_all, _ = read_data(train_data_dir=data_all_path, key='all_data_1.pkl')
    data_all = data_all[0]
    data_all_loader = DataLoader(data_all, batch_size=1024, shuffle=False)

    # initialize several models to compute the lowest initial loss
    criterion = torch.nn.CrossEntropyLoss()
    best_model = None
    best_loss = float('inf')
    best_alpha = 0 if options['alpha'] == 0 else options['alpha']
    compute_alpha = True if options['alpha'] == 0 else False
    start = time.time()

    for i_init in range(options['n_init']):
        # randomly initialize a new model
        random_model = choose_model(options)
        i_loss, i_alpha = eval_hessian(random_model, data_all_loader, criterion, options['device'], compute_alpha)
        # save the best model so far
        if i_loss < best_loss:
            best_loss = i_loss
            best_model = random_model

        if compute_alpha:
            best_alpha = max(best_alpha, i_alpha)

    best_model_init = get_flat_params_from(best_model).detach()
    options['alpha'] = best_alpha
    options['model_init'] = best_model_init
    print(f'>>> The estimate of constant alpha is {best_alpha}.')

    # read the data
    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)
    # load the fine-tuning dataset
    if options['ft_dataset'] == 'mnist-m':
        ft_train_loader, ft_test_loader = get_loader_mnist_m('./data/mnist_m', options['ft_dataset'], options['ft_batch_size'], num_workers=16)
    
    elif options['ft_dataset'] == 'svhn':
        ft_train_loader, ft_test_loader = get_loader_svhn('svhn', options['ft_batch_size'], num_workers=16)
    else:
        raise Exception("Did not find target dataset!")



    # record the metrics for different trails
    fl_test_acc = [0.0] * options['repeat']
    model_source_only_test_acc = [0.0] * options['repeat']
    model_ft_test_acc = [0.0] * options['repeat']
    model_target_only_test_acc = [0.0] * options['repeat']
    model_random_test_acc = [0.0] * options['repeat']
    
    for repeat_i in range(options['repeat']):
        print('\n' + '*' * 120 + '\n')
        print(f"repeat:{repeat_i+1}/{options['repeat']}")
        # Set seeds
        np.random.seed(repeat_i + options['seed'])
        torch.manual_seed(repeat_i + 10 + options['seed'])
        print(f"using torch seed {repeat_i + 10 + options['seed']}")
        if options['gpu']:
            torch.cuda.manual_seed_all(repeat_i + 100 + options['seed'])

        # create unique id for saving files
        uid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        print(f'uid: {uid}')

        # Call appropriate trainer
        model_path = f"./models/{options['model']}_{dataset_name}_{uid}.pt"
        print(f'FL pretrained model will be saved at {model_path}')
        trainer = trainer_class(options, all_data_info, model_path)
        fl_test_acc[repeat_i] = trainer.train()

        if not options['noft']:
            # FL training finish here, save the latest server model
            flat_model_params = trainer.latest_model
            torch.save(flat_model_params, model_path)
            flat_model_params = torch.load(model_path)

            # Initialize new models
            model_source_only = choose_model(options)  # baseline: lower bound (f on source, g on source)
            model_ft = choose_model(options)  # baseline: standard fine-tune (f on source, g on target)
            # model_target_only = choose_model(options)  # baseline: upper bound (f on target, g on target)
            # model_random = choose_model(options)  # baseline: lower bound (f random, g on target)

            # Assign model params
            set_flat_params_to(model_source_only, flat_model_params)
            set_flat_params_to(model_ft, flat_model_params)

            # Now model is set with flat_model_params
            # Start fine-tuning below
            # First, freeze all but last k fc layers
            # freeze(model_random, options['last_k'])
            freeze(model_ft, options['last_k'])

            checkpoint_prefix = f'./models/ft_checkpoints/{uid}_'
            # # Train model_target_only
            # print('>>> Training model_target_only')
            # _, model_target_only_results = ft_train(model_target_only, options, options['device'], ft_train_loader, ft_test_loader, checkpoint_prefix + 'model_target_only.pt')

            # fine-tuning
            print('>>> Training model_ft')
            _, model_ft_results = ft_train(model_ft, options, options['device'], ft_train_loader, ft_test_loader, checkpoint_prefix + 'model_ft.pt')

            # # fine-tuning random model
            # print('>>> Training model_random')
            # _, model_random_results = ft_train(model_random, options, options['device'], ft_train_loader, ft_test_loader, checkpoint_prefix + 'model_random.pt')

            # evaluate model_source_only
            print('>>> Evaluating model_source_only')
            model_source_only = model_source_only.to(options['device'])
            model_source_only_results = [0., 0., 0., 0.]
            model_source_only_results[0], model_source_only_results[1] = eval(model_source_only, options['device'],
                                                                              ft_train_loader, criterion=criterion)
            model_source_only_results[2], model_source_only_results[3] = eval(model_source_only, options['device'],
                                                                              ft_test_loader, criterion=criterion)

            # print(f'model_target_only: {model_target_only_results}')
            print(f'model_ft: {model_ft_results}')
            # print(f'model_random: {model_random_results}')
            print(f'model_source_only: {model_source_only_results}')
            
            # model_target_only_test_acc[repeat_i] = model_target_only_results[-1]
            model_ft_test_acc[repeat_i] = model_ft_results[-1]
            # model_random_test_acc[repeat_i] = model_random_results[-1]
            model_source_only_test_acc[repeat_i] = model_source_only_results[-1]

    
    
    print('fl_test_acc_mean', np.mean(fl_test_acc))
    print('model_source_only_test_acc_mean', np.mean(model_source_only_test_acc))
    print('model_ft_test_acc_mean', np.mean(model_ft_test_acc))
    # print('model_target_only_test_acc_mean', np.mean(model_target_only_test_acc))
    # print('model_random_test_acc_mean', np.mean(model_random_test_acc))
    
    final_metrics = {'fl_test_acc': fl_test_acc, 'model_source_only_test_acc': model_source_only_test_acc, 'model_ft_test_acc': model_ft_test_acc,
                    'model_target_only_test_acc': model_target_only_test_acc, 'model_random_test_acc': model_random_test_acc}
    
    with open(f'./result/mnist_all_data_1_equal_niid/{uid}.json', 'w') as ouf:
        json.dump(final_metrics, ouf)

if __name__ == '__main__':
    print("working!")
    main()
