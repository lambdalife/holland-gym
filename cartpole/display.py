import os, sys
import json
import gym

sys.path.append(os.getcwd())
from cartpole_agent import CartPoleAgent

def display():
    with open("cartpole/genomes.json", "r") as f:
        results = json.loads(f.readline())
        genome = results["results"][0][-1]

    agent = CartPoleAgent(genome)
    env = gym.make("CartPole-v1")
    obs = env.reset()

    done = False
    while not done:
        env.render()
        decision = agent.decide(obs)
        obs, reward, done, info = env.step(decision)

if __name__ == "__main__":
    display()