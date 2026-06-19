from mario_env import make_env
from agent import MarioAgent

def test():

    env = make_env()

    agent = MarioAgent("supermariobrosv0")
    agent.load()

    agent.epsilon = 0.0

    state = env.reset()
    done = False

    while not done :

        action = agent.select_action(state)

        state , reward , done , _ = env.step(action)

        env.render()
        print("Action:", action, "Reward:", reward)

if __name__ == "__main__" :
    test()

