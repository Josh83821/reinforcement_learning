# Add dynamic internal shape 

import torch
import torch.nn as nn
from torch.distributions import Normal

class a2cContinuousActor(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        mean = self.layer3(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def act(self, state):
        """
        Given a state, take action
        """
        mean, std = self.forward(state)
        
        dist = Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1) 
        
        return action, log_prob
    
    def get_name(self):
        return 'a2cContinuousActor'
        

class a2cContinuousCritic(nn.Module):
    
    def __init__(self, input_dim):
        
        super().__init__()
        
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

    def get_name(self):
        return 'a2cContinuousCritic'
        