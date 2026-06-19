from collections import deque
import random

class ReplayMemory:
    def __init__(self,maxlen):
        self.memory = deque(maxlen = maxlen)

    def push(
            self,
            state,
            action,
            reward,
            next_state,
            terminated
    ):
        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                terminated
            )
        )

    def sample(self,batch_size):
        return random.sample(self.memory , batch_size)
    
    def __len__(self):
        return len(self.memory)