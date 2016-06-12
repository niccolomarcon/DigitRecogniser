import numpy as np

class NeuralNetwork(object):
    def __init__(self, inputs, hiddens, outputs, lambda1 = 1.125, lambda2 = 2.25):
        # set some default vars
        self._inputNumber  = inputs
        self._hiddenNumber = hiddens
        self._outputNumber = outputs
        self.l1 = lambda1
        self.l2 = lambda2

        # initialize the weights
        self.W1 = np.random.randn(self._inputNumber, self._hiddenNumber)
        self.W2 = np.random.randn(self._hiddenNumber, self._outputNumber)

    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z) * (1. - self.sigmoid(z))

    def tanh(self, z):
        return np.tanh(z)

    def tanhPrime(self, z):
        return 1. - (self.tanh(z) ** 2)

    def forward(self, X):
        # first layer
        self.Z2 = np.dot(X, self.W1)
        self.A2 = self.tanh(self.Z2)

        # output layer
        self.Z3 = np.dot(self.A2, self.W2)
        self.Yh = self.sigmoid(self.Z3)

        return self.Yh

    def costFunction(self, X, Y):
        return 1. / 2. * sum((Y - self.forward(X)) ** 2)

    def costFunctionPrime(self, X, Y):
        self.costFunction(X, Y)

        delta3 = (-(Y - self.Yh)) * self.sigmoidPrime(self.Z3)
        dEdW2  = np.dot(self.A2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.tanhPrime(self.Z2)
        dEdW1  = np.dot(X.T, delta2)

        return dEdW1, dEdW2

    def train(self, X, Y, testX, testY, cycles = 4850, rate = 0.0017):
        self.TrainingPlot = []
        self.OverfitPlot = []

        for i in range(cycles):
            dw1, dw2 = self.costFunctionPrime(X, Y)
            self.W1 = self.W1 - (rate * (dw1 + self.W1 * self.l1))
            self.W2 = self.W2 - (rate * (dw2 + self.W2 * self.l2))
            self.TrainingPlot.append(sum(self.costFunction(X, Y)))
            self.OverfitPlot.append(sum(self.costFunction(testX, testY)))
