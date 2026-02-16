

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v1", render_mode="rgb_array")

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

state = env.reset()[0]
state = torch.from_numpy(state).float()


Actor = Actor_Model(observation_space, action_space)
actor_optimizer = torch.optim.Adam(Actor.parameters(), lr=0.001)

Critic = Critic_Model(observation_space)
critic_optimizer = torch.optim.Adam(Critic.parameters(), lr=0.001)

total_rewards = []
actor_losses = []

gamma = 0.99
logger_freq = 25

K = 5
epsilon = 0.2

c_entropy = 0.005

for ep in range(1000):

    #Initialize state and done
    state = env.reset()[0]
    state = torch.from_numpy(state).float()
    done = False

    log_probs = []
    q_vals = []
    rewards = deque()
    masks = deque()
    states = []
    actions = []
    entropies = []


    #While the emulator isn't done
    while not done:

        action, log_prob = Actor.act(torch.unsqueeze(state, dim = 0))

        q_val = Critic(state)
        
        states.append(state)
        actions.append(action)

        #Take actions and observe reward, etc.
        state, reward, terminated, truncated, _ = env.step(action.item())
        state = torch.from_numpy(state).float()

        #If emulator done terminate loop
        if truncated or terminated: 
            done = True

        #Save observed transition
        rewards.append(torch.tensor(reward))
        #entropies.append(entropy)
        log_probs.append(log_prob)
        q_vals.append(q_val)
        masks.append(torch.tensor([1-done], dtype=torch.float))

    total_rewards.append(int(sum(rewards).detach()))
    final_q = Critic(state)
    

    returns = calc_Gs(final_q, masks, rewards, gamma)
    returns = torch.cat(returns).detach()

    #entropies = torch.cat(entropies)
    old_log_prob = torch.cat(log_probs)
    q_vals = torch.cat(q_vals)

    advantage = returns - q_vals
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        
    for i in range(K):
        
        # Step 2: Train policy for K epochs
        new_log_prob, entropy = Actor.evaluate(torch.stack(states), torch.tensor(actions))  # Policy is being updated every epoch
        ratios = torch.exp(new_log_prob - old_log_prob.detach())
        
        # Clipped surrogate objective
        surr1 = ratios * advantage.detach() 
        
        
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantage.detach()
        actor_loss = (-torch.min(surr1, surr2) - c_entropy * entropy).mean()
        

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
    

    critic_loss = (advantage**2).mean()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    #actor_losses.append(actor_loss.detach())



    if ep%logger_freq == 0:
        print(ep, mean(total_rewards[-logger_freq:]))
s