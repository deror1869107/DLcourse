import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def dataset():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])

class NN:
    def __init__(self):
        self.in_size = 2
        self.hidden_size = 2
        self.out_size = 1
        # hidden layer weights
        self.wh = np.random.randn(self.hidden_size, self.in_size) / np.sqrt(self.hidden_size)
        # hidden layer bias
        self.bh = np.random.randn(self.hidden_size)
        # hidden layer outputs
        self.oh = np.zeros(self.hidden_size) 
        # output layer weights
        self.wo = np.random.randn(self.out_size, self.hidden_size) / np.sqrt(self.out_size)
        # output layer bias
        self.bo = np.random.randn(self.out_size)
        # output layer outputs
        self.out = np.zeros(self.out_size) 

    def forward(self, data):
        self.oh = np.zeros(self.hidden_size)
        self.out = np.zeros(self.out_size)

        for i in range(self.hidden_size):
            for j in range(self.in_size):
                self.oh[i] += self.wh[i][j] * data[j]
            self.oh[i] += self.bh[i]
            self.oh[i] = sigmoid(self.oh[i])

        for i in range(self.out_size):
            for j in range(self.hidden_size):
                self.out[i] += self.wo[i][j] * self.oh[j]
            self.out[i] += self.bo[i]
            self.out[i] = sigmoid(self.out[i])

        return self.out

    def backward(self, data, target, out, learning_rate):
        output_delta = np.zeros(self.out_size)

        for i in range(self.out_size):
            error = target[i] - out[i]
            output_delta[i] = error * -1 * d_sigmoid(out[i])

        for i in range(self.out_size):
            for j in range(self.hidden_size):
                self.wo[i][j] -= output_delta[i] * self.oh[j] * learning_rate
            self.bo[i] -= output_delta[i] * learning_rate

        hidden_delta = np.zeros(self.hidden_size)

        for i in range(self.hidden_size):
            error = 0.0
            for j in range(self.out_size):
                error += output_delta[j] * self.wo[j][i]
            hidden_delta[i] = error * d_sigmoid(self.oh[i])

        for i in range(self.hidden_size):
            for j in range(self.in_size):
                self.wh[i][j] -= hidden_delta[i] * data[j] * learning_rate
            self.bh[i] -= hidden_delta[i] * learning_rate

def main():
    data, target = dataset()
    epochs = 100000
    learning_rate = 1e-2
    np.random.seed(0)
    nn = NN()

    for epoch in range(epochs):
        for d, t in zip(data, target):
            out = nn.forward(d)
            nn.backward(d, [t], out, learning_rate)

        if epoch % 1000 == 0:
            loss = 0
            for d, t in zip(data, target):
                loss += (nn.forward(d)[0] - t) ** 2

            print("Epoch {}: Loss {}".format(epoch, loss))

if __name__ == '__main__':
    main()
