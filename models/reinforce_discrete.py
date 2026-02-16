# Add dynamic internal shape 

import torch
import torch.nn as nn
from torch.distributions import Categorical
    
class reinforceDiscreteNet(nn.Module):

    def __init__(self, input_dim, output_dim, mid_dims):
        
        super().__init__()
        
        self.first = nn.Linear(input_dim, mid_dims[0])

        self.layers = []
        for i in range(len(mid_dims)-1):
            layer = nn.Linear(mid_dims[i], mid_dims[i+1])
            self.layers.append(layer)
        self.last = nn.Linear(mid_dims[-1], output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 0)
        
    def get_name(self):
        return self.model_name
    
    def forward(self, x):
        
        x = self.relu(self.first(x))
        
        for layer in self.layers:
            x = self.relu(layer(x))

        return self.softmax(self.last(x))
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float()
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def get_name(self):
        return 'reinforceDiscreteNet'
