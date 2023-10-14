from src.utils.torch_utils import set_flat_params_to
from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch


criterion = torch.nn.CrossEntropyLoss()


class FedAvgTLTrainer(BaseTrainer):
    """
    Scheme I and Scheme II, based on the flag of self.simple_average
    """
    def __init__(self, options, dataset, checkpoint_path):
        model = choose_model(options)
        # set the model parameters to the best so far
        set_flat_params_to(model, options['model_init'])
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        self.opt_lr = options['opt_lr']
        worker = LrdWorker(model, self.optimizer, options)
        super(FedAvgTLTrainer, self).__init__(options, dataset, worker=worker)
        self.prob = self.compute_prob()
        self.alpha = options['alpha']
        self.checkpoint_path = checkpoint_path
        self.early_stopping = options['early_stopping']

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        best_train_loss = float('inf')
        best_train_acc = 0
        best_test_loss = float('inf')
        best_test_acc = 0
        patience = 0
        
        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)
            
            # check for early stopping after we evaluate the loss on training data
            if self.metrics.loss_on_train_data[round_i] < best_train_loss:
                # save the model to the given path
                torch.save(self.latest_model, self.checkpoint_path)
                best_train_loss = self.metrics.loss_on_train_data[round_i]
                best_train_acc = self.metrics.acc_on_train_data[round_i]
                best_test_loss = self.metrics.loss_on_eval_data[round_i]
                best_test_acc = self.metrics.acc_on_eval_data[round_i]
                patience = 0
            else:
                patience += 1
            
            if patience == self.early_stopping and self.early_stopping != 0:
                print(f"Training early stopped. Model saved at {self.checkpoint_path}.")
                break
            
            if self.clients_per_round < len(self.clients):
                # subsample K clients prop to data size
                if self.simple_average:
                    selected_clients, repeated_times = self.select_clients_with_prob(seed=round_i)
                else:
                    selected_clients = self.select_clients(seed=round_i)
                    repeated_times = None
            else:
                # use all clients
                selected_clients = [self.clients[i] for i in range(len(self.clients))]
                repeated_times = [1] * len(self.clients)

            # lr are determined at base class
            solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns, repeated_times=repeated_times)
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)
        
        # final check in case the latest model is the best one
        if self.metrics.loss_on_train_data[self.num_round] < best_train_loss:
            torch.save(self.latest_model, self.checkpoint_path)
            best_train_loss = self.metrics.loss_on_train_data[self.num_round]
            best_train_acc = self.metrics.acc_on_train_data[self.num_round]
            best_test_loss = self.metrics.loss_on_eval_data[self.num_round]
            best_test_acc = self.metrics.acc_on_eval_data[self.num_round]

        # Save tracked information
        self.metrics.write()
        
        return best_train_loss, best_train_acc, best_test_loss, best_test_acc

    def compute_prob(self):
        probs = []
        for c in self.clients:
            probs.append(len(c.train_data))
        return np.array(probs)/sum(probs)

    def select_clients_with_prob(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        index = np.random.choice(len(self.clients), num_clients, p=self.prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                repeated_times[-1] += 1
        return select_clients, repeated_times

    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        if self.simple_average:
            repeated_times = kwargs['repeated_times']
            assert len(solns) == len(repeated_times)
            for i, (num_sample, local_solution) in enumerate(solns):
                averaged_solution += local_solution * repeated_times[i]
            averaged_solution /= self.clients_per_round
        else:
            for num_sample, local_solution in solns:
                averaged_solution += num_sample * local_solution
            averaged_solution /= self.all_train_data_num
            averaged_solution *= (100/self.clients_per_round)

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

