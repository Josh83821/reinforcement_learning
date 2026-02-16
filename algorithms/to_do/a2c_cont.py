
        
        



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

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n


Actor = Actor_Model(observation_space, action_space)
actor_optimizer = torch.optim.Adam(Actor.parameters(), lr=0.001)

Critic = Critic_Model(observation_space)
critic_optimizer = torch.optim.Adam(Critic.parameters(), lr=0.001)

gamma = 0.99
logger_freq = 100

total_rewards = []
actor_losses = []


for ep in range(1001):

    #Initialize state and done
    state = env.reset()[0]
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
        state, reward, terminated, truncated, _ = env.step(action)
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

    # standardization of the returns is employed to make training more stable
    eps = np.finfo(np.float32).eps.item()


    # eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    returns = torch.cat(returns).detach()

    log_probs = torch.cat(log_probs)
    q_vals = torch.cat(q_vals)


    advantage = returns - q_vals
    #returns = (returns - returns.mean()) / (returns.std() + eps)

    actor_loss = - (log_probs * advantage.detach()).mean()
    critic_loss = (advantage**2).mean()

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()

    actor_losses.append(actor_loss.detach())

    if ep%logger_freq == 0:
        print(ep, mean(total_rewards[-logger_freq:]))
