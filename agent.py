import numpy as np


class Agent:
    def __init__(self, genome, dimensions):
        """
        Genome should have genes named "w{i}" where i starts at 0 and goes up to one less than the number of layers in the Neural Net
        """
        self.layers = []
        for i in range(len(dimensions) - 1):
            layer = np.array(genome.get(f"w{i}")).reshape(dimensions[i + 1], dimensions[i])
            self.layers.append(layer)

    def forward(self, vector):
        for layer in self.layers:
            vector = self.sigmoid(np.matmul(layer, vector))
        return vector

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
