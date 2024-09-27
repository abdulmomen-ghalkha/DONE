import copy

import torch
import os

from algorithms.edges.edgeDONE import edgeDONE
# from algorithms.edges.edgeSeOrder2 import edgeSeOrder2
from algorithms.edges.edgeFiOrder import edgeFiOrder
from algorithms.edges.edgeDANE import edgeDANE
from algorithms.edges.edgeNew import edgeNew
from algorithms.edges.edgeGD import edgeGD
from algorithms.edges.edgeFEDL import edgeFEDL
from algorithms.edges.edgeNewton import edgeNewton
from algorithms.edges.edgeAvg import edgeAvg
from algorithms.edges.edgeGT import edgeGT
from algorithms.edges.edgePGT import edgePGT
from algorithms.edges.edgeSophia import edgeSophia
from algorithms.edges.edgefedprox import edgeFedProx

from algorithms.server.serverbase import ServerBase
from utils.model_utils import read_data, read_edge_data
import numpy as np

# Implementation for Central Server
class Server(ServerBase):
    def __init__(self, experiment, device, dataset, algorithm, model, batch_size, learning_rate, alpha, eta, L,
                 num_glob_iters,
                 local_epochs, optimizer, num_edges, times):
        super().__init__(experiment, device, dataset, algorithm, model[0], batch_size, learning_rate, alpha, eta, L,
                         num_glob_iters,
                         local_epochs, optimizer, num_edges, times)

        # Initialize data for all  edges
        data = read_data(dataset, read_optimal_weights=False)

        self.optimal_weights = None
        self.optimal_loss_unreg = None  # Unregularized loss
        self.optimal_loss_reg = None  # Regularized loss with parameter L
        if data[-1] is not None:
            # Synthetic dataset: save the optimal weights for comparison later
            self.optimal_weights = data[-2]
            self.optimal_loss_unreg = data[-1]
            self.optimal_loss_reg = (self.L / 2) * (np.linalg.norm(data[-1]) ** 2)



        total_edges = len(data[0])
        self.alpha_users_G = torch.ones((total_edges)).to(self.device)
        self.alpha_server_G = torch.as_tensor(1).to(self.device)
        self.alpha_users_H = torch.ones((total_edges)).to(self.device)
        self.alpha_server_H = torch.as_tensor(1).to(self.device)
        self.SNR_db_total = 25

        self.device = device
        #torch.set_default_device(self.device)

        
        # Done 23 20 21 24
        # To be done 23 25 30 35 40
        for i in range(total_edges):

            id, train, test = read_edge_data(i, data, dataset)

            if (algorithm == "DONE"):
                edge = edgeDONE(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                                optimizer)

            if (algorithm == "FirstOrder"):
                edge = edgeFiOrder(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L,
                                   local_epochs, optimizer)

            if (algorithm == "DANE"):
                edge = edgeDANE(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                                optimizer)

            if algorithm == "New":
                edge = edgeNew(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                               optimizer)

            if algorithm == "GD":
                edge = edgeGD(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                              optimizer)

            if algorithm == "FedAvg":
                edge = edgeAvg(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                               optimizer)

            if (algorithm == "FEDL"):
                edge = edgeFEDL(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                                optimizer)

            if (algorithm == "Newton"):
                edge = edgeNewton(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L,
                                  local_epochs, optimizer)

            if (algorithm == "GT"):
                edge = edgeGT(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                              optimizer)
            if (algorithm == "PGT"):
                edge = edgePGT(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                               optimizer)
            if (algorithm == "Sophia"):
                edge = edgeSophia(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L,
                                  local_epochs,
                                  optimizer)
            if (algorithm == "Sophia_OTA"):
                edge = edgeSophia(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L,
                                  local_epochs,
                                  optimizer)
            if algorithm == "FedProx":
                edge = edgeFedProx(device, id, train, test, model, batch_size, learning_rate, alpha, eta, L, local_epochs,
                               optimizer)



            self.edges.append(edge)
            self.total_train_samples += edge.train_samples

        print("Number of edges / total edges:", num_edges, " / ", total_edges)

    def send_grads(self):
        assert (self.edges is not None and len(self.edges) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for edge in self.edges:
            edge.set_grads(grads)

    def send_dt(self):
        for edge in self.edges:
            edge.set_dt(self.dt)

    def train(self):
        loss = []
        if (self.algorithm == "FirstOrder"):
            # All edge will eun GD or SGD to obtain w*
            for edge in self.edges:
                edge.train(self.local_epochs)

            # Communication rounds
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()  # still evaluate on the global model
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)

                for edge in self.selected_edges:
                    edge.update_direction()

                self.aggregate_parameters()

        elif self.algorithm == "DANE":
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
                # Caculate gradient to send to server for average
                for edge in self.edges:
                    edge.get_full_grad()

                self.aggregate_grads()

                # receive average gradient form server 
                self.send_grads()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs)

                self.aggregate_parameters()

        elif self.algorithm == "New":
            for glob_iter in range(1, self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()

                for edge in self.selected_edges:
                    edge.train(self.local_epochs)
                self.aggregate_parameters()
            self.save_results()
            self.save_model()

        elif self.algorithm == "GD" or self.algorithm == "FedAvg":
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)

                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)
                self.aggregate_parameters()

        elif self.algorithm == "DONE":  # Second Order method
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")

                # recive parameter from server
                self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                # Caculate gradient to send to server for average
                for edge in self.selected_edges:
                    edge.get_full_grad()

                # self.aggregate_grads()
                self.aggregate_sub_grads()
                # receive average gradient form server 
                self.send_grads()

                # all note are trained 
                # self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)

                self.aggregate_parameters()

        elif self.algorithm == "FEDL":
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.send_grads()
                self.evaluate()

                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs)
                # self.selected_edges[0].train(self.local_epochs)
                self.aggregate_parameters()
                self.aggregate_grads()

        elif self.algorithm == "Newton":  # using Richardson
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
                # reset all direction after each global interation

                # Aggregate grads of client.
                for edge in self.edges:
                    edge.get_full_grad()
                self.aggregate_grads()

                self.dt = []
                self.total_dt = []
                for param in self.model.parameters():
                    self.dt.append(-param.grad)
                    self.total_dt.append(torch.zeros_like(param.data))

                # Richardson
                for r in range(self.local_epochs):
                    self.send_dt()
                    self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                    for edge in self.selected_edges:
                        edge.gethessianproduct(self.local_epochs, glob_iter)
                    self.aggregate_dt()

                self.aggregate_newton()

        elif self.algorithm == "Newton2":  # using inverse hessian
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()

                for edge in self.edges:
                    edge.get_full_grad()
                self.aggregate_grads()

                hess = self.aggregate_hessians()
                inverse_hess = torch.inverse(hess)
                grads = []
                for param in self.model.parameters():
                    grads.append(param.grad.clone().detach())
                grads_as_vector = torch.cat([-grads[0].data, -grads[1].data.view(1, 1)], 1)
                direction = torch.matmul(grads_as_vector, inverse_hess)
                weights_direction = direction[0, 0:-1].view(grads[0].shape)
                bias_direction = direction[0, -1].view(grads[1].shape)

                for param, d in zip(self.model.parameters(), [weights_direction, bias_direction]):
                    param.data.add_(self.alpha * d)

        elif self.algorithm == "GT" or self.algorithm == "PGT":
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                # recive parameter from server
                self.send_parameters()
                self.evaluate()
                # Caculate gradient to send to server for average
                for edge in self.edges:
                    edge.get_full_grad()

                self.aggregate_grads()
                # receive average gradient form server 
                self.send_grads()

                # all note are trained 
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)

                self.aggregate_parameters()

        elif (self.algorithm == "Sophia"):
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total number of parameters: {param_count}")

            # Communication rounds
            for glob_iter in range(self.num_glob_iters):

                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                # All edge will eun GD or SGD to obtain w*
                if glob_iter != 0:
                    self.evaluate()
                self.send_parameters()
                # self.evaluate()  # still evaluate on the global model

                for edge in self.edges:
                    edge.train(self.local_epochs, glob_iter)

                grads = self.aggregate_grads_sophia()

                hess = self.aggregate_hessians_sophia()

                winrate = 0
                for i, param in enumerate(self.model.parameters()):
                    with torch.no_grad():
                        ratio = (grads[i].abs() / (self.eta * self.batch_size * hess[i] + 1e-15)).clamp(None, 1)
                        # param_count += np.equal(ratio.numpy(), 1.0).reshape(-1).shape[0]
                        winrate += np.mean(np.equal(ratio.cpu().numpy(), 1.0).reshape(-1)) * \
                                   np.equal(ratio.cpu().numpy(), 1.0).reshape(-1).shape[0]
                        param.mul_(1 - self.L)
                        step_size_neg = - self.learning_rate
                        param.addcmul_(grads[i].sign(), ratio, value=step_size_neg)
                        # print(grads[i].sign())
                print(f"Win rate = {1 - (winrate / param_count)}")

        elif (self.algorithm == "Sophia_OTA"):
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total number of parameters: {param_count}")

            # Communication rounds
            for glob_iter in range(self.num_glob_iters):

                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                # All edge will eun GD or SGD to obtain w*
                if glob_iter != 0:
                    self.evaluate()
                self.send_parameters()
                # self.evaluate()  # still evaluate on the global model

                for edge in self.edges:
                    edge.train(self.local_epochs, glob_iter)

                grads = self.aggregate_grads_sophia_OTA()

                hess = self.aggregate_hessians_sophia_OTA()

                winrate = 0
                for i, param in enumerate(self.model.parameters()):
                    with torch.no_grad():
                        ratio = (grads[i].abs() / (self.eta * self.batch_size * hess[i] + 1e-15)).clamp(None, 1)
                        # param_count += np.equal(ratio.numpy(), 1.0).reshape(-1).shape[0]
                        winrate += np.mean(np.equal(ratio.cpu().numpy(), 1.0).reshape(-1)) * \
                                   np.equal(ratio.cpu().numpy(), 1.0).reshape(-1).shape[0]
                        param.mul_(1 - self.L)
                        step_size_neg = - self.learning_rate
                        param.addcmul_(grads[i].sign(), ratio, value=step_size_neg)
                        # print(grads[i].sign())
                print(f"Win rate = {1 - (winrate / param_count)}")
        elif (self.algorithm == "FedProx"):
            for glob_iter in range(self.num_glob_iters):
                if (self.experiment):
                    self.experiment.set_epoch(glob_iter + 1)
                print("-------------Round number: ", glob_iter, " -------------")
                if glob_iter != 0:
                    self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)
                self.aggregate_parameters()


        self.save_results()
        self.save_model()

    def weights_difference(self, weights=None, optimal_weights=None):
        """
        Calculate the norm of w - w*, the difference between the current weights
        and the optimal weights of the dataset.
        """
        if weights is None:
            weights = list(self.model.parameters())[0].data.clone().detach().flatten().numpy()
        if optimal_weights is None:
            optimal_weights = self.optimal_weights
        if weights.shape != optimal_weights.shape:
            weights = weights.T
        return np.linalg.norm(weights - optimal_weights)

    def regularize(self, model=None):
        model = self.model if model is None else model
        reg = 0
        for param in model.parameters():
            if param.requires_grad:
                reg += param.norm() ** 2
        return (self.L / 2) * reg

    def losses_difference(self, loss, optimal_loss=None, regularize=True):
        """
        Calculate f(w) - f(w*), the difference between the evaluation function
        at the current weights and at the optimal weights.
        """
        if optimal_loss is None:
            if regularize:
                optimal_loss = self.optimal_loss_reg
            else:
                optimal_loss = self.optimal_loss_unreg
        return loss - optimal_loss

    def aggregate_hessians(self):
        aggregated_hessians = None
        total_samples = 0
        for i, edge in enumerate(self.edges):
            hess = edge.send_hessian()
            total_samples += edge.train_samples
            if aggregated_hessians is None:
                aggregated_hessians = hess
            else:
                aggregated_hessians.add_(hess)
        return aggregated_hessians / (i + 1 + 1e-6)

    def aggregate_hessians_sophia_OTA(self):
        # Wireless settings
        Nc = 6000
        param_num_list = [param.numel() for param in self.model.parameters()]
        Nd = sum(param_num_list)
        Ns = np.ceil(Nd / Nc)
        h_thr = 1e-2
        SNR_db = self.SNR_db_total
        SNR = 10 ** (SNR_db / 10)
        P_t = 1e-3

        total_indicators = torch.zeros((Nd, 1)).to(self.device)

        aggregated_hess = torch.zeros((Nd, 1)).to(self.device)

        with torch.no_grad():
            for i in range(0, Nd, Nc):
                transmittable_list = []
                H_amp_list = []
                slice_end = min(i + Nc, Nd)

                for j, edge in enumerate(self.edges):
                    transmittable = np.ones((Nc, 1))
                    H = (1 / np.sqrt(2)) * np.random.randn(Nc, 1) + 1j * (1 / np.sqrt(2)) * np.random.randn(Nc, 1)
                    H_amp = np.real(np.conjugate(H) * H)
                    transmittable[H_amp < h_thr] = 0.0
                    H_amp = np.multiply(H_amp, transmittable)
                    H_amp_list.append(copy.deepcopy(torch.as_tensor(H_amp)).to(self.device))
                    transmittable_list.append(torch.as_tensor(copy.deepcopy(transmittable)).to(self.device))
                    # Transmitted signal
                    # print(f"{i} : {slice_end},    {min(Nc, Nd - i)}")
                    # Calculate the average power for each client
                    P_sum_nofactor = torch.sum(torch.nan_to_num(torch.multiply(
                        torch.div(torch.square(edge.h[i:slice_end, :]),
                                  torch.as_tensor(H_amp_list[j][:min(Nc, Nd - i), :])),
                        torch.as_tensor(transmittable_list[j][:min(Nc, Nd - i), :])), posinf=0.0, neginf=0.0))

                    # Number of Parameters to be transmitted
                    d = torch.sum(torch.as_tensor(transmittable_list[j][:min(Nc, Nd - i), :]))
                    P_avg_user1 = torch.div(P_sum_nofactor, d)

                    # Transmit the user alpha
                    self.alpha_users_H[j] = torch.nan_to_num(torch.div(d * P_t, P_sum_nofactor), posinf=0.0, neginf=-0.0)


                # Calculate the server side alpha
                self.alpha_server_H = torch.min(self.alpha_users_H)
                self.alpha_users_H[:] = self.alpha_server_H

                for j, edge in enumerate(self.edges):
                    # Ensure the last slice doesn't go out of bounds

                    S_n = torch.sqrt(self.alpha_server_H) * torch.multiply(torch.nan_to_num(edge.h[i:slice_end, :], posinf=0.0, neginf=-0.0).detach().clone(),torch.as_tensor(transmittable_list[j][:min(Nc, Nd - i),:]))


                    # print(f"User: {j}, alpha: {alpha_users[j]}")

                    aggregated_hess[i:slice_end, :] += torch.multiply(
                        transmittable_list[j][:min(Nc, Nd - i), :],
                        S_n)
                    total_indicators[i:slice_end, :] += transmittable_list[j][:min(Nc, Nd - i), :]

                    # Create a list of flattened tensors
                    # grad_list_flat = [param.detach().clone().view(-1) for param in edge.m]

                # Average the model parameter
                P_avg = torch.div(torch.square(aggregated_hess[i:slice_end, :]),
                                  total_indicators[i:slice_end, :]).to(self.device)
                term_a = torch.sqrt(torch.nan_to_num(torch.div(P_avg, (2 * self.alpha_server_H * SNR)), posinf=0.0, neginf=0.0)).to(self.device)
                noise_term = torch.randn((min(Nc, Nd - i), 1)).to(self.device)
                P_noise = torch.nan_to_num(torch.multiply(term_a, noise_term), posinf=0.0, neginf=0.0)
                aggregated_hess[i:slice_end, :] = torch.add(
                    torch.nan_to_num(torch.div(aggregated_hess[i:slice_end, :],
                                               torch.sqrt(self.alpha_server_H) * total_indicators[i:slice_end, :]),
                                     posinf=0.0, neginf=0.0), P_noise)
                #print(
                #    f"H_Noise: {torch.sum(torch.square(P_noise)) / Nd},  indicators: {torch.mean(total_indicators[slice_end - 10:slice_end - 1, :])}, Pavg: {torch.sum(P_avg)}, alpha: {self.alpha_server_H}, P_Noise: {torch.mean(P_noise)}")
                #print(aggregated_hess[i:slice_end, :])

            layer_wise_hess = [torch.nan_to_num(x.view(y.shape), posinf=0.0, neginf=-0.0).detach().clone() for x, y in zip(torch.split(aggregated_hess,
                                                                            split_size_or_sections=param_num_list,
                                                                            dim=0),
                                                                self.model.parameters())]

        return layer_wise_hess

    def aggregate_grads_sophia_OTA(self):
        # Wireless settings
        Nc = 6000
        param_num_list = [param.numel() for param in self.model.parameters()]
        Nd = sum(param_num_list)
        Ns = np.ceil(Nd / Nc)
        h_thr = 1e-2
        SNR_db = self.SNR_db_total
        SNR = 10 ** (SNR_db / 10)
        P_t = 1e-3

        total_indicators = torch.zeros((Nd, 1)).to(self.device)

        aggregated_grad = torch.zeros((Nd, 1)).to(self.device)

        with torch.no_grad():
            for i in range(0, Nd, Nc):
                transmittable_list = []
                H_amp_list = []
                slice_end = min(i + Nc, Nd)

                for j, edge in enumerate(self.edges):
                    transmittable = np.ones((Nc, 1))
                    H = (1 / np.sqrt(2)) * np.random.randn(Nc, 1) + 1j * (1 / np.sqrt(2)) * np.random.randn(Nc, 1)
                    H_amp = np.real(np.conjugate(H) * H)
                    transmittable[H_amp < h_thr] = 0.0
                    H_amp = np.multiply(H_amp, transmittable)
                    H_amp_list.append(copy.deepcopy(torch.as_tensor(H_amp)).to(self.device))
                    transmittable_list.append(torch.as_tensor(copy.deepcopy(transmittable)).to(self.device))
                    # Transmitted signal
                    # print(f"{i} : {slice_end},    {min(Nc, Nd - i)}")
                    # Calculate the average power for each client
                    P_sum_nofactor = torch.sum(torch.nan_to_num(torch.multiply(
                        torch.div(torch.square(edge.m[i:slice_end, :]),
                                  torch.as_tensor(H_amp_list[j][:min(Nc, Nd - i), :])),
                        torch.as_tensor(transmittable_list[j][:min(Nc, Nd - i), :])), posinf=0.0, neginf=0.0))

                    # Number of Parameters to be transmitted
                    d = torch.sum(torch.as_tensor(transmittable_list[j][:min(Nc, Nd - i), :]))
                    P_avg_user1 = torch.div(P_sum_nofactor, d)

                    # Transmit the user alpha
                    self.alpha_users_G[j] = torch.nan_to_num(torch.div(d * P_t, P_sum_nofactor), posinf=0.0, neginf=0.0)

                # Calculate the server side alpha
                self.alpha_server_G = torch.min(self.alpha_users_G)
                self.alpha_users_G[:] = self.alpha_server_G

                for j, edge in enumerate(self.edges):
                    # Ensure the last slice doesn't go out of bounds

                    S_n = torch.sqrt(self.alpha_server_G) * torch.multiply(edge.m[i:slice_end, :].detach().clone(),
                                                                               transmittable_list[j][:min(Nc, Nd - i),
                                                                               :])

                    # print(f"User: {j}, alpha: {alpha_users[j]}")

                    #print(S_n.get_device(), transmittable_list[j].get_device(), aggregated_grad.get_device())

                    aggregated_grad[i:slice_end, :] += torch.multiply(
                        transmittable_list[j][:min(Nc, Nd - i), :],
                        S_n)
                    total_indicators[i:slice_end, :] += transmittable_list[j][:min(Nc, Nd - i), :]

                    # Create a list of flattened tensors
                    # grad_list_flat = [param.detach().clone().view(-1) for param in edge.m]

                # Average the model parameter
                P_avg = torch.div(torch.square(aggregated_grad[i:slice_end, :]),
                                  total_indicators[i:slice_end, :]).to(self.device)
                term_a = torch.sqrt(torch.nan_to_num(torch.div(P_avg, (2 * self.alpha_server_G * SNR)), posinf=0.0, neginf=0.0)).to(self.device)
                noise_term = torch.randn((min(Nc, Nd - i), 1)).to(self.device)
                P_noise = torch.nan_to_num(torch.multiply(term_a, noise_term), posinf=0.0, neginf=0.0)
                aggregated_grad[i:slice_end, :] = torch.add(
                    torch.nan_to_num(torch.div(aggregated_grad[i:slice_end, :],
                                               torch.sqrt(self.alpha_server_G) * total_indicators[i:slice_end, :]),
                                     posinf=0.0, neginf=0.0), P_noise)
                #print(
                #    f"G_Noise: {torch.sum(torch.square(P_noise))  / Nd},  indicators: {torch.mean(total_indicators[slice_end - 10:slice_end - 1, :])}, Pavg: {torch.sum(P_avg)}, alpha: {self.alpha_server_G}")
                #print(aggregated_grad[i:slice_end, :])

            layer_wise_grad = [x.view(y.shape).detach().clone() for x, y in zip(torch.split(aggregated_grad,
                                                                           split_size_or_sections=param_num_list,
                                                                           dim=0),
                                                               self.model.parameters())]

        return layer_wise_grad





    def aggregate_hessians_sophia(self):

        param_num_list = [param.numel() for param in self.model.parameters()]
        Nd = sum(param_num_list)
        aggregated_hess = torch.zeros((Nd, 1)).to(self.device)
        
        with torch.no_grad():




            for j, edge in enumerate(self.edges):
                aggregated_hess += edge.h


            aggregated_hess = torch.div(aggregated_hess, len(self.edges))


            layer_wise_hess = [x.view(y.shape).detach().clone() for x, y in zip(torch.split(aggregated_hess,
                                                                            split_size_or_sections=param_num_list,
                                                                            dim=0),
                                                                self.model.parameters())]

        return layer_wise_hess

    def aggregate_grads_sophia(self):
        param_num_list = [param.numel() for param in self.model.parameters()]
        Nd = sum(param_num_list)
        aggregated_grad = torch.zeros((Nd, 1)).to(self.device)


        with torch.no_grad():

            for j, edge in enumerate(self.edges):

                # print(f"User: {j}, alpha: {alpha_users[j]}")
                aggregated_grad += edge.m
            

            aggregated_grad = torch.div(aggregated_grad, len(self.edges))
                


            layer_wise_grad = [x.view(y.shape).detach().clone() for x, y in zip(torch.split(aggregated_grad,
                                                                           split_size_or_sections=param_num_list,
                                                                           dim=0),
                                                               self.model.parameters())]

        return layer_wise_grad
