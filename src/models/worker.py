from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch
from torch.autograd.functional import *
from PIL import Image
import numpy as np

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
        
        if options["reg_J_flag"]:
            self.latest_J0 = None
            self.lbd_J = options['lbd_reg_J']
        
        if options["model"] == '2nn' or options["model"] == 'logistic':
            self.flat_data = True
        else:
            self.flat_data = False

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
                x, y = x.type(torch.float).cuda(), y.cuda()
                # x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_Jacobian(self):
        grad = torch.tensor([])
        if self.gpu:
            grad = grad.cuda()
        
        for mod in self.model.children():
            grad = torch.cat(grad,mod.grad)
        return grad


    
    def local_train(self, train_dataloader, reg_J_flag, **kwargs):
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
            for batch_idx, (x, y) in enumerate(train_dataloader):
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.type(torch.float).cuda(), y.cuda()
                    # x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                if torch.isnan(pred.max()):
                    from IPython import embed
                    embed()

                loss = criterion(pred, y) 
                if reg_J_flag:
                    J_local = self.get_Jacobian()
                    loss += self.lbd_J * torch.norm(J_local - self.latest_J0)
                loss.backward()
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
        return local_solution, return_dict, J_local

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                # x = Image.fromarray(x)
                x = self.flatten_data(x)
                if self.gpu:
                    
                    x, y = x.type(torch.float).cuda(), y.cuda()

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
        super(LrdWorker, self).__init__(model, optimizer, options)

    def get_flat_Jacobian_from(self,loss):
        grads = []
        for param in self.model.parameters():
            grads.append(param.grad.view(-1))
            # grads.append(jacobian(loss, param).view(-1))
            flat_grads = torch.cat(grads)
        return flat_grads
    
    def get_flat_Jacobian_from_avoid_backward(self,loss):
        grads = []
        for param in self.model.parameters():
            
            grads.append(torch.autograd.grad(loss, param.data.requires_grad_(), create_graph = True)[0].view(-1))
            flat_grads = torch.cat(grads)
        return flat_grads
    
    def local_train(self, train_dataloader, reg_J_flag, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch*10):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.type(torch.float).cuda(), y.cuda()
                # x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            

            if not reg_J_flag:
                loss.backward()
                J_local = self.get_flat_Jacobian_from(loss)

            else:
               
                J_local = self.get_flat_Jacobian_from_avoid_backward(loss)
                
                self.optimizer.zero_grad()
                loss = criterion(pred, y) + self.lbd_J * torch.norm(J_local - self.latest_J0)
                loss.backward()
                J_check = self.get_flat_Jacobian_from(loss)
                print(f"new grad: \n{J_check}")
            
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
        return local_solution, return_dict, J_local

    def local_train_reg_J(self, train_dataloader, lbd_reg_J, gradnorm, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch*10):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.type(torch.float).cuda(), y.cuda()
                # x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss_0 = criterion(pred, y)
            
            grad_in_tensor = self.get_flat_grads(train_dataloader)
            # return grad_in_tenser.cpu().detach().numpy()
            
            # for c in self.clients:
            #     (num, client_grad), stat = c.solve_grad()
            #     local_grads.append(client_grad)
            #     num_samples.append(num)
            #     global_grads += client_grad * num
            #     global_grads /= np.sum(np.asarray(num_samples))
            
            local_gradnorm = torch.norm(grad_in_tensor)
            loss_reg_J = (gradnorm - local_gradnorm)**2
            
            loss = loss_0 + lbd_reg_J * loss_reg_J
               
            
            
            self.optimizer.zero_grad()
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
        return local_solution, return_dict


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
                x, y = x.type(torch.float).cuda(), y.cuda()
                # x, y = x.cuda(), y.cuda()
        
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
