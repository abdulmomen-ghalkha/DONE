import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *
from torch.functional import F

class edgeSophia(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        self.pre_params = []
        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()#nn.NLLLoss()
        #print(alpha, eta)
        self.optimizer = SophiaG(self.model.parameters(), lr=learning_rate, betas=alpha, rho=eta,
                                 weight_decay=L, maximize=False, capturable=False)
        # Keep track of local hessians and exp_avg
        self.param_num_list = [param.numel() for param in self.model.parameters()]
        self.total_param_num =sum(self.param_num_list)

        self.m = torch.zeros((self.total_param_num, 1))
        self.h = torch.zeros((self.total_param_num, 1))
        self.k = 5



    def train(self, epochs, glob_iter):
        # Only update once time
        self.model.train()
        for i, (X, y) in zip(range(1), self.trainloaderfull):
            X, y = X.to(self.device), y.to(self.device)

            #self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss(logits, y)
            loss.backward()
            self.optimizer.step(bs=self.batch_size)
            self.optimizer.zero_grad(set_to_none=True)

            if glob_iter % self.k == 0:

                # update hessian EMA
                logits = self.model(X)
                samp_dist = torch.distributions.Categorical(logits=logits)
                y_sample = samp_dist.sample()
                loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                loss_sampled.backward()
                self.optimizer.update_hessian()
                self.optimizer.zero_grad(set_to_none=True)
                m, h = self.optimizer.get_m_h()
                self.m = torch.cat([param.detach().clone().view(-1) for param in m], dim=0).reshape(self.total_param_num, 1)
                self.h = torch.cat([param.detach().clone().view(-1) for param in h], dim=0).reshape(self.total_param_num, 1)
            else:
                m, h = self.optimizer.get_m_h()
                self.m = torch.cat([param.detach().clone().view(-1) for param in m], dim=0).reshape(
                    self.total_param_num, 1)
                self.h = torch.cat([param.detach().clone().view(-1) for param in h], dim=0).reshape(
                    self.total_param_num, 1)

    def send_grad(self):
        return copy.deepcopy(self.m)


    def send_hessian(self):
        return copy.deepcopy(self.h)
