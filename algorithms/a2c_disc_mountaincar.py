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





class a2cDiscMountainCar(algParent):
    
    def __init__(self, env, learning_model, hypers):
        super().__init__(env, hypers, learning_model)
        self.alg = 'a2cDiscMountainCar'
    
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
            count = 0

            log_probs = []
            q_vals = []
            rewards = deque()
            masks = deque()

            #While the emulator isn't done
            while not done:
                count+=1

                action, log_prob = Actor.act(state)

                q_val = Critic(state)

                #Take actions and observe reward, etc.
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state).float()
                # reward = orig_reward

                '''# Reward moving up a hill
                if next_state[0] > state[0] and action == 2:
                    reward += 1.5
                elif next_state[0] < state[0] and action == 0:
                    reward +=1.5'''
                
                '''# Reward increasing speed at any given point
                if abs(next_state[1]) > abs(state[1]):
                    reward += 2'''
                
                '''accelerating = abs(next_state[1]) < abs(state[1])
                rhs = state[0] > 0
                acc_r = action == 2
                acc_l = action == 0'''

                # Reward accelerating opposite direction when slowing down on hill
                '''if abs(next_state[1]) < abs(state[1]): 
                    if next_state[]'''
                
                '''if accelerating:
                    if rhs and acc_r:
                        reward += 1.5
                        # print('REWARD')
                    if not rhs and acc_l:
                        reward += 1.5
                        # print('REWARD')
                else:
                    if rhs and acc_l:
                        reward += 1.5
                        # print('REWARD')
                    if not rhs and acc_r:
                        reward += 1.5
                        # print('REWARD')'''
                
                
                additional_reward = 2*abs(next_state[0])
                reward += additional_reward

                state = next_state

                #If emulator done terminate loop
                if truncated: 
                    done = True
                elif terminated:
                    done = True
                    reward += 100

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
                print(f'Epoch length: {count}')
                print(ep, mean(total_rewards[-logger_freq:]))

        self.loss_lists.append(actor_losses)
        self.rewards_lists.append(total_rewards)


