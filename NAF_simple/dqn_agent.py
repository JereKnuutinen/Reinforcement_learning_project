#from distutils.command.config import config
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
sys.path.insert(0, os.path.abspath("../.."))
from common import helper as h

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class NAF(nn.Module):
    def __init__(self, state_shape, action_size,
                 batch_size=256, hidden_dims=256, gamma=0.99, lr=1e-3, grad_clip_norm=1.0, tau=0.005):
        super(NAF, self).__init__()
        # base layers 1-4 with skip connections
        print(state_shape)
        combined_size = state_shape + hidden_dims
        # layer 1
        self.l1 = nn.Linear(in_features = state_shape, out_features = hidden_dims)
        #self.bn1 = nn.BatchNorm1d(hidden_dims)
        # layer 2
        self.l2 = nn.Linear(hidden_dims, hidden_dims)
        #self.bn2 = nn.BatchNorm1d(hidden_dims)
        # layer 3
        #self.l3 = nn.Linear(combined_size, hidden_dims)
        #self.bn3 = nn.BatchNorm1d(hidden_dims)
        # layer 4
        #self.l4 = nn.Linear(combined_size, hidden_dims)
        #self.bn4 = nn.BatchNorm1d(hidden_dims)        
        # layer 1-4 are shared with all the heads
        # heads:
        # mu : maximum action, apply Tanh in forward
        self.mu = nn.Linear(hidden_dims, action_size)
        # lower triangular entries of L, apply Tanh in forward
        self.L_entries = nn.Linear(hidden_dims, int(action_size * (action_size + 1) / 2))
        # value function (scalar)
        self.V = nn.Linear(hidden_dims, 1)
        # store constructor parameters   
        self.action_size = action_size
        self.state_dim = state_shape
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.counter = 0
        
    def forward(self, state, action=None, min_action=-1, max_action=1, bnorm = False):
        # base layers feedforward
        # sometimes state [1x11] and sometimes [256x11]
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        
        # maximum action
        mu = torch.tanh(self.mu(x))

        l_entries = torch.tanh(self.L_entries(x))
        
        Value = self.V(x)
        # create lower triangular matrix L
        #L = torch.zeros((self.state_dim, self.action_size, self.action_size))
        L = torch.zeros((state.shape[0], self.action_size, self.action_size))
        triang_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)
        L[:, triang_indices[0], triang_indices[1]] = l_entries
        L.diagonal(1,2).exp_() # exponentiate diagonal elements
        
        P = L * torch.transpose(L, 2, 1) # or : P = L * L.transpose
        
        Advantage = None
        Q = None        
        if action!=None:
            a1 = (action - mu).unsqueeze(1)
            a2 = (action - mu).unsqueeze(-1)
            Advantage = (-0.5 * a1 @ P @ a2).squeeze(-1)
            #A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - mu).transpose(2, 1), P), (action.unsqueeze(-1) - mu))).squeeze(-1)
            Q = Advantage + Value
        
        # sample action
        new_action = torch.distributions.MultivariateNormal(mu.squeeze(-1), torch.inverse(P)).sample()
        #new_action = torch.distributions.MultivariateNormal(mu, P.inverse()).sample()
        new_action = new_action.clamp(min_action, max_action)
        
        return mu, Q, new_action, Advantage, Value
        

class DQNAgent(object):
    def __init__(self, state_shape, n_actions,
                 batch_size=32, hidden_dims=64, gamma=0.98, lr=1e-3, grad_clip_norm=1000, tau=0.001):
        self.n_actions = n_actions
        self.state_dim = state_shape[0]

        self.policy_net = NAF(self.state_dim, n_actions, batch_size, hidden_dims)
        self.target_net = copy.deepcopy(self.policy_net)
        #self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        
        self.counter = 0

    # modified from exercise 4
    def update(self, buffer):
        """ One gradient step, update the policy net."""
        self.counter += 1
        batch = buffer.sample(self.batch_size, device=device)
        # calculate the q(s,a)
        mu, Q, new_action, Advantage, Value = self.policy_net.forward(batch.state, batch.action)
        # calculate V target
        with torch.no_grad():
            mu_t, Q_t, new_action_t, Advantage_t, Value_t = self.target_net(batch.next_state)
            V_target = batch.reward + (self.gamma * Value_t * batch.not_done)

        loss = F.smooth_l1_loss(Q, V_target)
        self.optimizer.zero_grad()
        loss.backward()
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm, error_if_nonfinite=False)
        self.optimizer.step()
        ########## You code ends here #########

        # update the target network
        h.soft_update_params(self.policy_net, self.target_net, self.tau)
        
        return {'loss': loss.item(),
                'num_update': self.counter}

    @torch.no_grad()
    def get_action(self, state):
        # TODO:  Task 3: implement epsilon-greedy action selection
        ########## You code starts here #########
        if state.ndim == 1:
            state = state[None] # add batch dimension
        state = torch.tensor(state, ).to(torch.float32).to(device) # conver state to tensor and put it to device
        _, _, new_action, _, _ = self.policy_net(state)
        return new_action


    def save(self, fp):
        path = fp/'dqn.pt'
        torch.save({
            'policy': self.policy_net.state_dict(),
            'policy_target': self.target_net.state_dict()
        }, path)

    def load(self, fp):
        path = fp/'dqn.pt'
        d = torch.load(path)
        self.policy_net.load_state_dict(d['policy'])
        self.target_net.load_state_dict(d['policy_target'])