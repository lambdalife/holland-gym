from agent import Agent


class CartPoleAgent(Agent):
    dimensions = (4, 4, 1)

    def __init__(self, genome):
        super().__init__(genome, self.dimensions)

    def decide(self, input_vector):
        decision_vector = self.forward(input_vector)
        return int(round(decision_vector[0]))
