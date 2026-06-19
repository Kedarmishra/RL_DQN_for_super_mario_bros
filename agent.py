import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import os
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
RUNS_DIR = "runs"
os.makedirs("runs", exist_ok = True)


class MarioAgent:

    def __init__(self , param_set):

        self.param_set = param_set

        with open("parameters.yaml" , "r") as f:
            all_param_set = yaml.safe_load(f)
        params = all_param_set[param_set]

        self.env_id = params["env_id"]
        
        self.epsilon = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.gamma = params["gamma"]
        self.alpha = params["alpha"]

        self.replay_memory_size = params["replay_memory_size"]
        self.batch_size = params["mini_batch_size"]
        self.memory = ReplayMemory(self.replay_memory_size)

        self.network_sync_rate = params["network_sync_rate"]
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR , f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR , f"{self.param_set}.pt")
        self.device = device

        self.num_actions = len(SIMPLE_MOVEMENT)
        self.burn_in = params["burn_in"]
        self.learn_every = params["learn_every"]
        

        self.policy_dqn = DQN(
            (4,84,84),
            self.num_actions
        ).to(device)

        self.target_dqn = DQN(
            (4,84,84),
            self.num_actions
        ).to(device)

        self.target_dqn.load_state_dict(
            self.policy_dqn.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy_dqn.parameters(),
            lr = self.alpha
        )

    def select_action(self , state):

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1 )
        
        with torch.no_grad():
            state = torch.as_tensor(
                np.array(state),
                dtype = torch.float32,
                device = self.device
            ).unsqueeze(0)

            q_values = self.policy_dqn(state)

            action = torch.argmax(q_values).item()
        
        return action
        
    
    def decay_epsilon(self) :
        if self.epsilon > self.epsilon_min:
             self.epsilon *= self.epsilon_decay
        
        self.epsilon = max(
            self.epsilon,
            self.epsilon_min
        )

    def remember(
            self,
            state,
            action,
            reward,
            next_state,
            terminated
    ):
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            terminated
        )
    
    def train(self) :

        if len(self.memory) < self.batch_size:
            return
        
        mini_batch = self.memory.sample(
            self.batch_size
        )

        states , actions , rewards , next_states , terminations = zip(
            *mini_batch
        )

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        terminations = torch.FloatTensor(terminations).to(self.device)

        current_q = self.policy_dqn(states).gather(
            dim = 1 , 
            index = actions.unsqueeze(1)
        ).squeeze()
        with torch.no_grad():

            next_actions = self.policy_dqn(next_states).argmax(
                dim = 1, 
                keepdim = True
            )
            
            next_q = self.target_dqn(next_states).gather(
                dim = 1,
                index = next_actions
            ).squeeze()

            target_q = rewards + (1-terminations) * self.gamma * next_q
        loss = self.loss_fn(
            current_q,
            target_q
        )
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.decay_epsilon()

    def sync_target_network(self):

        self.target_dqn.load_state_dict(
            self.policy_dqn.state_dict()
        )
    
    def save(self) :

        torch.save(
            self.policy_dqn.state_dict(),
            self.MODEL_FILE
        )

    def load(self) :
        self.policy_dqn.load_state_dict(
            torch.load(
                self.MODEL_FILE,
                map_location = self.device
            )
        )

        self.target_dqn.load_state_dict(
            self.policy_dqn.state_dict()
        )




        