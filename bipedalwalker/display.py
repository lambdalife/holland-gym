import os, sys
import json
import gym

sys.path.append(os.getcwd())
from bipedalwalker_agent import BipedalWalkerAgent


def display():
    with open("bipedalwalker/genomes.json", "r") as f:
        results = json.loads(f.readline())
        genome = results["results"][0][-1]

    agent = BipedalWalkerAgent(genome)
    env = gym.make("BipedalWalker-v2")
    obs = env.reset()

    done = False
    while not done:
        env.render()
        decision = agent.decide(obs)
        obs, reward, done, info = env.step(decision)


if __name__ == "__main__":
    display()
