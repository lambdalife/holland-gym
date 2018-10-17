import os, sys
import random
import gym
from holland import Evolver, library

sys.path.append(os.getcwd())
from bipedalwalker_agent import BipedalWalkerAgent

gym.logger.set_level(40)  # env gives numpy type warnings


def fitness_function(genome):
    agent = BipedalWalkerAgent(genome)
    env = gym.make("BipedalWalker-v2")

    score = 0
    n_trials = 5

    for i in range(n_trials):
        obs = env.reset()
        env.seed(i)
        done = False
        while not done:
            decision = agent.decide(obs)
            obs, reward, done, info = env.step(decision)
            score += reward

    return score / n_trials


genome_params = {
    "w0": {
        "type": "[float]",
        "size": BipedalWalkerAgent.dimensions[1] * BipedalWalkerAgent.dimensions[0],
        "initial_distribution": lambda: random.random() * 2 - 1,
        "crossover_function": library.get_point_crossover_function(n_crossover_points=2),
        "mutation_function": library.get_gaussian_mutation_function(sigma=0.5),
        "mutation_rate": 0.1,
    },
    "w1": {
        "type": "[float]",
        "size": BipedalWalkerAgent.dimensions[2] * BipedalWalkerAgent.dimensions[1],
        "initial_distribution": lambda: random.random() * 2 - 1,
        "crossover_function": library.get_point_crossover_function(n_crossover_points=2),
        "mutation_function": library.get_gaussian_mutation_function(sigma=0.5),
        "mutation_rate": 0.1,
    },
}

selection_strategy = {"pool": {"top": 10}}

evolver = Evolver(fitness_function, genome_params, selection_strategy)

storage_options = {
    "genomes": {
        "should_record_genomes": True,
        "format": "json",
        "file_name": "genomes.json",
        "path": "bipedalwalker",
        "top": 1,
    }
}

final_pop = evolver.evolve(stop_conditions={"target_fitness": 300}, storage_options=storage_options)


best_genome = final_pop[-1][1]

print("Best Genome:")
print(best_genome)
