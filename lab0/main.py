import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def dataset():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])

class FC:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.w = np.random.randn(self.out_size, self.in_size)
        self.x = np.zeros((self.out_size, self.in_size))
        self.b = np.random.randn(self.out_size)
        self.out = np.zeros(self.out_size)
        self.prev = None

    def __call__(self, x):
        if type(x) == FC:
            self.prev = x
            input = x.out
        else:
            input = x

        self.forward(input)
        return self

    def forward(self, x):
        self.x = x
        self.out = np.zeros(self.out_size)

        self.out = np.matmul(self.w, self.x.T) + self.b
        self.out = sigmoid(self.out)

        return self.out

    def backward(self, prev_delta, learning_rate):
        delta = prev_delta * d_sigmoid(self.out)
        next_delta = np.matmul(delta, self.w)

        self.w -= np.multiply.outer(delta, self.x) * learning_rate
        self.b -= delta * learning_rate

        if self.prev:
            self.prev.backward(next_delta, learning_rate)

class Loss:
    def __init__(self, output, target, lr):
        self.output = output
        self.target = target
        self.lr = lr

    def backward(self):
        self.output.backward(self.output.out - self.target, self.lr)

class Net:
    def __init__(self):
        self.fc1 = FC(2, 2)
        self.fc2 = FC(2, 4)
        self.fc3 = FC(4, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def main():
    data, target = dataset()
    epochs = 40000
    learning_rate = 1e-1
    np.random.seed(0)
    model = Net()

    # train
    for epoch in range(epochs):
        for d, t in zip(data, target):
            out = model.forward(d)
            loss = Loss(out, t, learning_rate)
            loss.backward()

        if epoch % 100 == 0:
            loss = 0
            for d, t in zip(data, target):
                loss += np.sum((model.forward(d).out - t) ** 2)

            print("Epoch {}: Loss {}".format(epoch, loss))

    # eval
    for d in data:
        print(d, model.forward(d).out)

if __name__ == '__main__':
    main()
