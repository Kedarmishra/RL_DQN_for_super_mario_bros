import itertools
from mario_env import make_env
from agent import MarioAgent
import numpy as np

def train():

    env = make_env()

    agent = MarioAgent(
        "supermariobrosv0"
    )

    step_count = 0 

    best_reward = float("-inf")

    for episode in itertools.count():

        state= env.reset()

        done = False
        episode_reward = 0
        while not done :

            action = agent.select_action(
                state
            )

            next_state , reward ,done, _ = env.step(
                action
            )
            

            agent.remember(
                state ,
                action , 
                reward ,
                next_state , 
                done
            )

            state = next_state
            step_count += 1

            if len(agent.memory) > agent.burn_in:

                if step_count % agent.learn_every == 0:
                    agent.train()

                if step_count % agent.network_sync_rate == 0 :
                    agent.sync_target_network()

            episode_reward += reward
        best_reward = max(best_reward , episode_reward)

        print(f"Episode {episode + 1} with epsilon {agent.epsilon:4f}  and reward {episode_reward : 1f} and best_reward {best_reward : 1f}")
        
        if episode % 100 == 0 :
            agent.save()

if __name__ == "__main__" :
    train()


                
