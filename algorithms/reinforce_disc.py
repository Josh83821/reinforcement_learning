from .parent_alg_class import algParent
import torch.nn as nn
import random
import torch
import matplotlib.pyplot as plt
import copy
import numpy as np
from statistics import mean 

from collections import deque
from torch.distributions import Categorical



def calc_Gs(rewards, gamma):
    Gs = deque()
    prev = 0
    for _ in range(len(rewards)):
        R = rewards.pop()
        prev = R + gamma * prev
        Gs.appendleft(prev)
    return Gs

def calc_loss(disc_rew, log_probs):
    Loss = 0
    for _ in range(len(disc_rew)):
        
        Loss -= disc_rew.pop() * log_probs.pop()
    return Loss

def rolling_avg(arr, dist):
    new = []
    for i in range(len(arr)-dist):
        bef = i
        after = i + 5
        new_val = arr[bef:after].mean()
        new.append(new_val)
    
    return new


class reinforceDisc(algParent):


    def __init__(self, env, learning_model, hypers):
        super().__init__(env, hypers)
        self.learning_model = learning_model
        

    def single_test(self, test_hypers, sound):
        
        Q, optimizer = self.learning_model.instantiate_model_and_optimizer()

        total_rewards = []
        losses = []

        rewards = deque()
        log_probs = deque()


        epochs = self.hypers['epochs']
        gamma = self.hypers['gamma']
        logger_freq = test_hypers['logger_freq']


        for ep in range(no_of_runs):

            #Initialize state and done
            state = self.env.reset()[0]
            done = False

            #While the emulator isn't done
            while not done:

                action, log_prob = Q.act(state)

                #Take actions and observe reward, etc.
                state, reward, terminated, truncated, _ = self.env.step(action)

                #If emulator done terminate loop
                if truncated or terminated: 
                    done = True

                #Save observed transition
                rewards.append(reward)
                log_probs.append(log_prob)

            total_rewards.append(sum(rewards))

            disc_rew = calc_Gs(rewards, gamma)

            ## standardization of the returns is employed to make training more stable
            eps = np.finfo(np.float32).eps.item()

            ## eps is the smallest representable float, which is
            # added to the standard deviation of the returns to avoid numerical instabilities
            returns = torch.tensor(disc_rew)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            disc_rew = deque(returns)

            loss = calc_loss(disc_rew, log_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().item())

            if ep%logger_freq == 0 and sound:
                print(ep, mean(total_rewards[-logger_freq:]))
                
        self.loss_lists.append(losses)
        self.rewards_lists.append(total_rewards)
    
