import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
from torchvision import transforms as T
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class SkipFrame(gym.Wrapper):

    def __init__(self , env , skip) :
        super().__init__(env)
        self.skip = skip

    def step(self , action):
        total_reward = 0.0

        for _ in range(self.skip) :

            obs , reward ,done, info = self.env.step(action)
           
            total_reward += reward

            if done :
                break

        return obs , total_reward ,done , info
    
class GrayScaleObservation(gym.ObservationWrapper) :

    def __init__(self , env) :
        super().__init__(env)

        obs_shape = self.observation_space.shape[:2]

        self.observation_space = Box(
            low = 0,
            high = 255,
            shape = obs_shape,
            dtype = np.uint8
        )

    def permute_orientation(self , observation) :

        observation = np.transpose(
            observation,
            (2,0,1)
        )

        observation = torch.tensor(
            observation.copy(),
            dtype = torch.float
        )

        return observation
    
    def observation(self , observation) :
         
         observation = self.permute_orientation(
             observation
         )

         transform = T.Grayscale()
         observation = transform(observation)

         return observation
    
class ResizeObservation(gym.ObservationWrapper) :

    def __init__(self , env , shape) :
        super().__init__(env)

        if isinstance(shape , int) :
            self.shape = (shape , shape)
        else :
            self.shape = tuple(shape)

        obs_shape = self.shape

        self.observation_space = Box(
            low = 0 ,
            high = 255,
            shape = obs_shape,
            dtype = np.uint8
        )

    def observation(self , observation) :

        transform = T.Compose(
            [
                T.Resize(self.shape)
            ]
        )

        observation = transform(observation).squeeze(0) / 255.0
        return observation
    
def make_env() :

    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0"
    )

    env = JoypadSpace(
        env , 
        SIMPLE_MOVEMENT
    )

    env = SkipFrame(
        env , 
        skip = 4
    )

    env = GrayScaleObservation(env)

    env = ResizeObservation(
        env ,
        shape = 84
    )
    
    env = FrameStack(
        env , 
        num_stack = 4
    )

    return env