import numpy as np
import torch
import time
from src.models.client import Client
from src.utils.worker_utils import Metrics
from src.models.worker import Worker


class BaseTrainer(object):
    def __init__(self, options, dataset, model=None, optimizer=None, name='', worker=None):
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')

        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        print('>>> Weigh updates by {}'.format(
            'simple average' if self.simple_average else 'sample numbers'))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()
        self.opt_lr = options['opt_lr']
        self.reg_max = options['reg_max']
        self.lr = options['lr']

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Do not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            1. seed: random seed
            2. num_clients: number of clients to select; default 20
                note that within function, num_clients is set to min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs

        if round_i == 0:
            # initialization
            norm_avg_grad_at_global_weight_last_round = None
            avg_grad_at_global_weight_last_round = None
            max_grad_norm_sq_at_global_weight_last_round = None
            max_grad_for_reg = None
            sqrt_max_grad_norm_sq_at_global_weight_last_round = None
        else:
            # calculate avg_grad_at_global_weight_last_round and its norm
            local_grads_at_global_weight = []
            local_grad_norms_sq_at_global_weight = []
            for i, c in enumerate(selected_clients, start=1):
                # Communicate the latest model
                c.set_flat_model_params(self.latest_model)
                c_grad_at_global_weight = c.worker.get_flat_grads(c.train_dataloader).detach()  # tensor at device
                local_grads_at_global_weight.append(c_grad_at_global_weight)
                local_grad_norms_sq_at_global_weight.append(torch.norm(c_grad_at_global_weight)**2)
            
            avg_grad_at_global_weight_last_round = torch.mean(torch.stack(local_grads_at_global_weight, dim=0), dim=0)  # J_p
            
            norm_avg_grad_at_global_weight_last_round = torch.norm(avg_grad_at_global_weight_last_round)  # \|J_p\|
            
            avg_grad_norm_sq_at_global_weight_last_round = torch.mean(torch.stack(local_grad_norms_sq_at_global_weight, dim=0), dim=0)  # 1/K * \sum \|J^(k)\|^2
            
            max_grad_norm_sq_at_global_weight_last_round = torch.max(torch.stack(local_grad_norms_sq_at_global_weight, dim=0))  # max \|J^(k)\|^2

            sqrt_max_grad_norm_sq_at_global_weight_last_round = torch.sqrt(max_grad_norm_sq_at_global_weight_last_round)  # max \|J^(k)\|

            max_grad_for_reg = avg_grad_at_global_weight_last_round * sqrt_max_grad_norm_sq_at_global_weight_last_round / norm_avg_grad_at_global_weight_last_round  # J_p * \|J^(k)\| / \|J_p\|

            var_grad_at_global = avg_grad_norm_sq_at_global_weight_last_round - norm_avg_grad_at_global_weight_last_round**2  # sigma^2

        if self.opt_lr and round_i != 0:
            new_lr = (1 / self.alpha) * (norm_avg_grad_at_global_weight_last_round**2 / avg_grad_norm_sq_at_global_weight_last_round)
            # in case the lr gets too small
            if new_lr > self.lr:
                self.optimizer.set_lr(new_lr)
        
        if round_i == 0:
            print(f'round {round_i}: local lr = {self.optimizer.get_current_lr()}')
        else:
            print(f'round {round_i}: local lr = {self.optimizer.get_current_lr()}, '
                  f'sq_norm_avg_grad = {norm_avg_grad_at_global_weight_last_round**2}, avg_sq_norm_grad = {avg_grad_norm_sq_at_global_weight_last_round}, '
                  f'max_norm_grad = {sqrt_max_grad_norm_sq_at_global_weight_last_round}, var_grad = {var_grad_at_global}')
        
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            if self.reg_max:
                soln, stat = c.local_train(last_round_avg_local_grad_norm=sqrt_max_grad_norm_sq_at_global_weight_last_round, 
                                           last_round_global_grad=max_grad_for_reg)
            else:
                soln, stat = c.local_train(last_round_avg_local_grad_norm=norm_avg_grad_at_global_weight_last_round,
                                           last_round_global_grad=avg_grad_at_global_weight_last_round)

            if self.print_result and False:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            for num_sample, local_solution in solns:
                averaged_solution += num_sample * local_solution
            averaged_solution /= self.all_train_data_num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        # # this step already sets the client model to be the latest model (i.e., server model)
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient, keep for future use
        # model_len = len(self.latest_model)
        # global_grads = np.zeros(model_len)
        # num_samples = []
        # local_grads = []
        # local_grads_norm_square = 0

        # for c in self.clients:
        #     (num, client_grad), stat = c.solve_grad()
        #     local_grads.append(client_grad)
        #     local_grads_norm_square += np.linalg.norm(client_grad) ** 2
        #     num_samples.append(num)
        #     global_grads += client_grad * num
        # global_grads /= np.sum(np.asarray(num_samples))
        # stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # # Measure the gradient difference
        # difference = 0.
        # for idx in range(len(self.clients)):
        #     difference += np.sum(np.square(global_grads - local_grads[idx]))
        # difference /= len(self.clients)
        # stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result and round_i % self.eval_every == 0:
            # print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
            #       ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
            #        round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
            #        stats_from_train_data['gradnorm'], difference, end_time-begin_time))
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  'Time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                   end_time-begin_time))
            print('=' * 102 + '\n')
        # return global_grads, local_grads_norm_square

    def test_latest_model_on_evaldata(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            print('=' * 102 + '\n')

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def local_test(self, use_eval_data=True):
        assert self.latest_model is not None
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}

        return stats
