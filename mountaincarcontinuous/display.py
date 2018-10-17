import os, sys
import json
import gym

sys.path.append(os.getcwd())
from mountaincarcontinuous_agent import MountainCarContinuousAgent


def display():
    with open("mountaincarcontinuous/genomes.json", "r") as f:
        results = json.loads(f.readline())
        genome = results["results"][0][-1]

    agent = MountainCarContinuousAgent(genome)
    env = gym.make("MountainCarContinuous-v0")
    obs = env.reset()

    done = False
    while not done:
        env.render()
        decision = agent.decide(obs)
        obs, reward, done, info = env.step(decision)


if __name__ == "__main__":
    display()
