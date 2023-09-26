from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch
import copy


criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.batch_size = options['batch_size']
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False
        if options['model'] == '2nn' or options['model'] == 'logistic':
            self.flat_data = True
        else:
            self.flat_data = False
        if 'reg_J' in options and options['reg_J']:
            self.reg_J_coef = options['reg_J_coef']
            self.alpha = options['alpha']
        else:
            self.reg_J_coef = 0.0
            self.alpha = 0.0
        
        self.reg_J_norm_coef = options['reg_J_norm_coef']
        self.reg_J_ind_coef = options['reg_J_ind_coef']
        
        # Setup local model and evaluate its statics
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

    @property
    def model_bits(self):
        return self.model_bytes * 8
    
    def flatten_data(self, x):
        if self.flat_data:
            current_batch_size = x.shape[0]
            return x.reshape(current_batch_size, -1)
        else:
            return x

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:            
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_flat_grads_from_data(self, x, y):
        self.optimizer.zero_grad()
        x = self.flatten_data(x)
        if self.gpu:
            x, y = x.cuda(), y.cuda()
        pred = self.model(x)
        loss = criterion(pred, y)
        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0
            # loss term for reg_J
            if self.reg_J_coef != 0:
                # TODO: think about the reg_J term for Worker class
                raise NotImplementedError
                latest_model_local_grad = self.get_flat_grads(train_dataloader).detach()

            for batch_idx, (x, y) in enumerate(train_dataloader):
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                if torch.isnan(pred.max()):
                    from IPython import embed
                    embed()

                loss = criterion(pred, y)
                loss.backward()

                # do not clip gradient norm in our case
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss/train_total,
                       "acc": train_acc/train_total}
        return_dict.update(param_dict)

        # in the end, we need to compute the gradient again on the local dataset
        local_grad = self.get_flat_grads(train_dataloader).detach()

        return local_solution, return_dict, local_grad

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        self.repeat_epoch = options['repeat_epoch']
        self.clip = options['clip']
        super(LrdWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, last_round_avg_local_grad_norm=None,  last_round_global_grad = None, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch * self.repeat_epoch):
            x, y = next(iter(train_dataloader))

            
            loss_reg_J = loss_reg_J_ind = loss_reg_J_norm = 0
            
            
            latest_model_local_grad = self.get_flat_grads_from_data(x, y)
            
            # loss term for reg_J
            if self.reg_J_coef != 0 and last_round_avg_local_grad_norm is not None:
                # print("Applying UB reg with coef = ", self.reg_J_coef)
                cur_lr = self.optimizer.get_current_lr()
                loss_reg_J = self.alpha * (cur_lr ** 2) / 2 * torch.norm(latest_model_local_grad) ** 2 - cur_lr * last_round_avg_local_grad_norm ** 2
            
            if self.reg_J_norm_coef!= 0 and last_round_avg_local_grad_norm is not None:
                # print("Applying J norm reg with coef = ", self.reg_J_norm_coef)
                local_gradnorm = torch.norm(latest_model_local_grad)
                loss_reg_J_norm = (last_round_avg_local_grad_norm - local_gradnorm)**2

            if self.reg_J_ind_coef != 0 and last_round_global_grad is not None:
                # print("Applying J reg with coef = ", self.reg_J_ind_coef)
                loss_reg_J_ind = torch.norm(last_round_global_grad - latest_model_local_grad)**2
            
            

            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y)
            
            if self.reg_J_coef != 0 and last_round_avg_local_grad_norm is not None: 
                loss += self.reg_J_coef * loss_reg_J
            
            if self.reg_J_norm_coef!= 0 and last_round_avg_local_grad_norm is not None:
                loss += self.reg_J_norm_coef * loss_reg_J_norm
            
            if self.reg_J_ind_coef!= 0 and last_round_global_grad is not None:
                loss += self.reg_J_ind_coef * loss_reg_J_ind
            
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)

        # in the end, we need to compute the gradient again on the local dataset
        local_grad = self.get_flat_grads(train_dataloader).detach()

        return local_solution, return_dict, local_grad


class LrAdjustWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrAdjustWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        m = kwargs['multiplier']
        current_lr = self.optimizer.get_current_lr()
        self.optimizer.set_lr(current_lr * m)
        
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch*10):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)
        
        self.optimizer.set_lr(current_lr)
        return local_solution, return_dict

    
class FedProxWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(FedProxWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        
        # record the snapshot of the global model parameters
        global_model_params = copy.deepcopy(self.model.state_dict())
        
        for i in range(self.num_epoch * 10):
            x, y = next(iter(train_dataloader))
            
            # compute proximal loss
            proximal_term = 0.0
            if self.reg_J_coef != 0:
                for w_key, w_t_key in zip(self.model.state_dict(), global_model_params):
                    proximal_term += (self.model.state_dict()[w_key] - global_model_params[w_t_key]).norm() ** 2

            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y) + self.reg_J_coef * proximal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)

        # in the end, we need to compute the gradient again on the local dataset
        local_grad = self.get_flat_grads(train_dataloader).detach()

        return local_solution, return_dict, local_grad