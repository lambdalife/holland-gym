from agent import Agent


class MountainCarContinuousAgent(Agent):
    dimensions = (2, 2, 1)

    def __init__(self, genome):
        super().__init__(genome, self.dimensions)

    def decide(self, input_vector):
        decision_vector = self.forward(input_vector)
        return decision_vector * 2 - 1
