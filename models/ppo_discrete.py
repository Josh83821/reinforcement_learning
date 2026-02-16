# Add dynamic internal shape 

import torch
import torch.nn as nn
from torch.distributions import Categorical

class ppoDiscreteActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        logits = self.layer3(x)
        return self.softmax(logits)

    def act(self, state):
        """
        Used during rollout to sample action.
        """
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)#, m.entropy()

    def evaluate(self, states, actions):
        """
        Used during PPO update.
        Given batch of states and actions, return log_probs and entropies.
        """
        probs = self.forward(states)
        m = Categorical(probs)
        log_probs = m.log_prob(actions)
        entropy = m.entropy()
        return log_probs, entropy
    
    def get_name(self):
        return 'ppoDiscreteActor'
        

    
class ppoDiscreteCritic(nn.Module):
    
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
        return 'ppoDiscreteCritic'
