import copy

import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

class edgeFedProx(Edgebase):
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
            self.loss = nn.CrossEntropyLoss()
        self.global_model = copy.deepcopy(model[0])
        for param in self.global_model.parameters():
            torch.nn.init.zeros_(param)

        self.optimizer =  torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_global(self, model):
        for old_param, new_param in zip(self.global_model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()


    def train(self, epochs, glob_iter):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            (X, y) = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            for w, w_g in zip(self.model.parameters(), self.global_model.parameters()):
                w.grad.data += self.alpha * (w.data - w_g.data)
            self.optimizer.step()

