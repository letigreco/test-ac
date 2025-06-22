import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

#Funtion to compute G
def discount_rewards(r, gamma):
    """
    Computation of return G for each  time-stamp and storage in a tensor
    @param r tensor of rewards
    @parm gamma discount factor
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]   #Computation of return G for each time-stamp
        discounted_r[t] = running_add   #Saving of returns in a tensor
    return discounted_r


class Policy(torch.nn.Module):  #Sub-class of NN PyTorch class
    def __init__(self, state_space, action_space, type_alg, layer_size=64):
        super().__init__()
        self.state_space = state_space   #Attribute: state space
        self.action_space = action_space   #Attribute: action space
        self.hidden = layer_size   #Attribute: number of nodes in hidden layers
        self.tanh = torch.nn.Tanh()   #Attribute: activation function
        self.type_alg = type_alg

        #Actor network
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        #Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)  #Treat sigma as a parameter to learn like the weights of the NN. A parameter to update, but independent from the state

        if type_alg == 2:
            #Critic network
            self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
            self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
            self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        #Forward in the actor network
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)   #Normalized policy outputs (probabilities of actions)

        if self.type_alg == 2:
            #Forward in the critic network
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            action_value = self.fc3_critic(x_critic)
            return normal_dist, action_value
        
        return normal_dist


class Value(torch.nn.Module):
    def __init__(self, state_space, action_space, layer_size=64):
        super().__init__()
        self.state_space = state_space   #Attribute: state space
        self.action_space = action_space   #Attribute: action space
        self.hidden = layer_size   #Attribute: number of nodes in hidden layers
        self.tanh = torch.nn.Tanh()   #Attribute: activation function

        #Critic network
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        #Forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        critic_value = self.fc3_critic_value(x_critic)
        
        return critic_value
    


class Agent(object):
    def __init__(self, type_alg, policy, value = None, device='cpu', baseline=0, gamma=0.99, alpha=0.9, optim_lr=1e-3):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=optim_lr)   #Optimization algorithm on the policy parameters
        if type_alg == 1:
            self.value = value.to(self.train_device)
            self.optimizer_value = torch.optim.Adam(value.parameters(), lr=optim_lr)   #Optimization algorithm on the value parameters

        self.gamma = gamma   #Discount factor
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.alpha = alpha #Weight for the actor loss in type_alg == 2, used to balance actor and critic losses
        self.baseline = baseline
        self.type_alg=type_alg
        self.I = 1 # gamma^t, used in type_alg == 2 


    def update_policy(self):
        """
        From trajectory to optimization step. Update policy params. All time-stamps simultaneously.
        """
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)   #Concatenate tensors in list along a new axis, move the new tensor on chosen devise and remove entra dimensions of size 1 from the end.
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)   #Create a 1-dimensional tensor from a list of bools

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
        
        if self.type_alg == 0:
            returns = discount_rewards(rewards, self.gamma)
            returns -= returns.mean()
            returns/= (returns.std()+1e-8)  #TODO Normalizzare prima o dopo aver messo la baseline?
            # Compute loss, gradients and step the optimizer
            loss_fn =-torch.mean(action_log_probs * (returns-self.baseline))
            self.optimizer.zero_grad()
            loss_fn.backward()   #Compute the gradients of the loss w.r.t. each parameter
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1)   #Bring gradient norm to 1 if bigger
            self.optimizer.step()   #Compute a step of the optimization algorithm

        elif self.type_alg == 1:
            value_states = self.value(states)
            advantage_term = rewards+self.gamma*self.value(next_states)*(1-done)-value_states

            actor_loss_fn = -action_log_probs*advantage_term.detach()
            
            critic_loss_fn = 1/2*(advantage_term.pow(2))
            # ACTOR: compute gradients and step the optimizer
            self.optimizer.zero_grad()
            actor_loss_fn.backward()   #Compute the gradients of the loss w.r.t. each parameter
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1)   #Bring gradient norm to 1 if bigger
            self.optimizer.step()   #Compute a step of the optimization algorithm
            # CRITIC: compute gradients and step the optimizer
            self.optimizer_value.zero_grad()
            critic_loss_fn.backward()   #Compute the gradients of the loss w.r.t. each parameter
            torch.nn.utils.clip_grad_norm_(self.value.parameters(),1)   #Bring gradient norm to 1 if bigger
            self.optimizer_value.step()   #Compute a step of the optimization algorithm

        elif self.type_alg == 2:

            # Compute boostrapped discounted return estimates
            _, state_values = self.policy(states)                # V(s_t)
            _, next_state_values = self.policy(next_states)      # V(s_{t+1})
            done = done.float()

            td_target = rewards + self.gamma * next_state_values * (1 - done)  # if done=1 → no bootstrapping
            td_target = td_target.detach()  # Detach from the graph to avoid backpropagation through the next state value
            
            td_error = td_target - state_values  # delta = R_t + gamma*V(s_{t+1}) - V(s_t)

            action_log_probs = action_log_probs
            actor_loss = -self.I * action_log_probs * td_error
            critic_loss = 1/2*(td_error.pow(2))   
                 
            self.optimizer.zero_grad()
            (self.alpha*actor_loss + (1-self.alpha)*critic_loss).backward()

            self.optimizer.step()

            self.I = self.I * self.gamma  # Update the I for the next step

        return 

    def reset_I(self):
        """
        Reset the I value to 1 for the next episode (type_alg == 2)
        """
        self.I = 1           
        
        return 
    

    def get_action(self, state, evaluation=False):
        """ 
        Computation of one step of the trajectory
        state -> action (3-d), action_log_densities
        @param state current state
        @param evaluation do not act, return distribution mean
        @return action following action in trajectory
        @return action_log_prob probability of chosen action (joint probability of the three action values)
        """
        x = torch.from_numpy(state).float().to(self.train_device)
        out = self.policy(x)

        if isinstance(out, tuple):
            normal_dist = out[0]  # ignore Critic if present (type_alg == 2)
        else:
            normal_dist = out
           
        if evaluation:  # Return mean
            return normal_dist.mean, None
        else:   # Sample from the distribution  (choose an action)
            action = normal_dist.sample()
            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob, normal_dist
        
    def get_probs(self, state, action):
        """ 
        Computation of action_log_prob given action
        @param state current state
        @param evaluation do not act, return distribution mean
        @return action following action in trajectory
        @return action_log_prob probability of chosen action (joint probability of the three action values)
        """
        x = torch.from_numpy(state).float().to(self.train_device)
        out = self.policy(x)

        if isinstance(out, tuple):
            normal_dist = out[0]  # ignore Critic if present (type_alg == 2)
        else:
            normal_dist = out
        # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
        action_log_prob = normal_dist.log_prob(action).sum()
        return action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """
        Save the step of the trajectory in the class attributes. Store all the trajectory steps together
        """
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

