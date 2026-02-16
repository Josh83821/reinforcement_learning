from .parent_alg_class import algParent
import torch
from statistics import mean 
from collections import deque
import numpy as np


def calc_Gs(final_q, masks, rewards, gamma):
    
    Gs = []
    prev = final_q
    for _ in range(len(rewards)):
        R = rewards.pop()
        mask = masks.pop()
        prev = R + gamma * prev * mask
        Gs.insert(0, prev)
    return Gs

def calc_loss(disc_rew, log_probs):
    Loss = 0
    length = len(disc_rew)
    for _ in range(length):
        
        Loss -= disc_rew.pop().detach() * log_probs.pop()
        
    return Loss/length





class a2cDisc(algParent):
    
    def __init__(self, env, learning_model, hypers):
        super().__init__(env, hypers, learning_model)
        self.alg = 'a2cDisc'
    
    def single_test(self, test_hypers, sound):
        
        Actor, actor_optimizer = self.learning_model.return_model('actor').instantiate_model_and_optimizer()
        Critic, critic_optimizer = self.learning_model.return_model('critic').instantiate_model_and_optimizer()

        gamma = self.hypers['gamma']
        epochs = self.hypers['epochs']
        logger_freq = test_hypers['logger_freq']

        total_rewards = []
        actor_losses = []
        critic_losses = []


        for ep in range(epochs):

            #Initialize state and done
            state = self.env.reset()[0]
            state = torch.from_numpy(state).float()
            done = False

            log_probs = []
            q_vals = []
            rewards = deque()
            masks = deque()

            #While the emulator isn't done
            while not done:

                action, log_prob = Actor.act(state)

                q_val = Critic(state)

                #Take actions and observe reward, etc.
                state, reward, terminated, truncated, _ = self.env.step(action)
                state = torch.from_numpy(state).float()

                #If emulator done terminate loop
                if truncated or terminated: 
                    done = True

                #Save observed transition
                rewards.append(torch.tensor(reward))
                log_probs.append(log_prob.unsqueeze(0))
                q_vals.append(q_val)
                masks.append(torch.tensor([1-done], dtype=torch.float))

            total_rewards.append(int(sum(rewards).detach()))
            final_q = Critic(state)

            returns = calc_Gs(final_q, masks, rewards, gamma)

            ## standardization of the returns is employed to make training more stable
            eps = np.finfo(np.float32).eps.item()


            ## eps is the smallest representable float, which is
            # added to the standard deviation of the returns to avoid numerical instabilities
            returns = torch.cat(returns).detach()

            log_probs = torch.cat(log_probs)
            q_vals = torch.cat(q_vals)


            advantage = returns - q_vals
            #advantage = (advantage - advantage.mean()) / (advantage.std() + eps)

            actor_loss = - (log_probs * advantage.detach()).mean()
            critic_loss = (advantage**2).mean()

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

            actor_losses.append(actor_loss.detach())
            critic_losses.append(critic_loss.detach())

            if ep%logger_freq == 0 and sound:
                print(ep, mean(total_rewards[-logger_freq:]))

        self.loss_lists.append(actor_losses)
        self.rewards_lists.append(total_rewards)


