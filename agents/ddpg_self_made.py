import sys, os
sys.path.insert(0, os.path.abspath(".."))
import copy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from common import helper as h
from common.buffer import ReplayBuffer
from torch.distributions import Normal

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]


class DDPG(object):
    def __init__(self, state_shape, action_dim, max_action, lr, gamma, tau, batch_size, buffer_size=1e6):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action
        self.pi = Policy(state_dim, action_dim, max_action).to(device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

        self.q = Critic(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
    

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transiton

        if self.buffer_ptr > self.random_transition: # update once have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info


    def _update(self,):
        batch = self.buffer.sample(self.batch_size, device=device)

        # TODO: Task 2
        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using h.soft_update_params() (See the DQN code)
        
        self.q_optim.zero_grad()
        Q_tar = batch.reward + (self.gamma * self.q_target.forward(batch.next_state, self.pi_target(batch.next_state)))*(batch.not_done)
        critic_loss = F.mse_loss(self.q.forward(batch.state, batch.action), Q_tar)
        critic_loss.backward()
        self.q_optim.step()
        
        
        self.pi_optim.zero_grad()
        actor_loss = self.q.forward(batch.state,self.pi.forward(batch.state)) # * self.pi.forward(batch.state)
        actor_loss = - torch.mean(actor_loss)
        actor_loss.backward()
        self.pi_optim.step()
        
        

        h.soft_update_params(self.q, self.q_target, self.tau)
        h.soft_update_params(self.pi, self.pi_target, self.tau)

        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}

    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        if self.buffer_ptr < self.random_transition: # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            expl_noise = 0.1 * self.max_action # the stddev of the expl_noise if not evaluation
            
            # TODO: Task 2
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action's shape is correct.
            # pass
            action = self.pi.forward(x)
            
            if evaluation==False:
                # add normal noise to every action output of policy net
                #normal_noise = Normal(0,expl_noise)
                action = action + (expl_noise * torch.randn(self.action_dim)).to(device) #normal_noise.sample(action.size()).to(device)
                


            ########## Your code ends here. ##########

        return action, {} # just return a positional value


    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    
    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        pass
    
    def save(self, filepath):
        pass