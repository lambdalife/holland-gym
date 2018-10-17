from agent import Agent


class BipedalWalkerAgent(Agent):
    dimensions = (24, 24, 4)

    def __init__(self, genome):
        super().__init__(genome, self.dimensions)

    def decide(self, input_vector):
        return self.forward(input_vector)
