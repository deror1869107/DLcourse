Deep Learning and Practice Lab 0
====

## Introduction

使用 Multi Fully Connected Layer Neural Network 、 Sigmoid Function 和 Back Propagation 訓練 XOR gate

## Experiment setups

### Sigmoid

在多層 Fully Connected Network 中，必須使用非線性且可微分的函數作為 Activation Function，讓多層 Fully Connected Network 不致於退化成單層。

這裡選擇的是 Sigmoid Function

$$ f(x) = \frac{1}{1 + e^{-x}} $$

微分式為

$$ f'(x) = f(x) * (1 - f(x)) $$

只需知道 Sigmoid Function 的輸出就可以算出微分的結果

而且在 Output 層時可以將結果轉到 (0, 1)，符合 XOR 的輸出

### Nerual Network

#### Unit

數學式為

$$ y = f(w^Tx + b) $$

其中

x 為輸入，是一向量

w 為 weights，是大小和輸入相同向量

b 為 bias，是實數

f 為 Activation Function

y 為輸出，是實數

#### Single Fully Connected Layer

是由許多 Units 組成的，擁有兩個 parameters，input size 和 output size，代表輸入和輸出向量大小。

數學形式為

$$ y = f(W^Tx + b) $$

其中

x 為輸入，是一向量

W 為 weights，是一個大小為 input size x output size 的矩陣

b 為 bias，是大小為 output size 的向量

f 為 Activation Function

y 為輸出，是大小為 output size 的向量

#### Multi Fully Connected Layer

![](https://i.imgur.com/sJXiAEY.png)

將前一層 Fully Connected Layer 的輸出接入下一層 Fully Connected Layer，就可以建立多層 Fully Connected Layer 的 Neural Network

Python 實現為

```python
def forward(self, x):
    out = np.matmul(self.w.T, x) + self.b
    out = sigmoid(out)
    return out
```

### Back Propagation

Neural Network 訓練的方法是更新 weights 來將 Cost Function 的值最小化

這裡選擇的 Cost Function 為

$$ J(\theta) = \frac{1}{2}(y - f(x;\theta))^2 $$

其中 x 為輸入，theta 為 weights，y 為整個 Neural Network 的輸出 

目標為

$$ \frac{{\delta}J(\theta)}{\delta\theta} = 0 $$

而 Back Propagation 就是要計算出 theta 的 gradient

$$ \nabla_\theta J(\theta) $$

#### 以一個 2x2 Hidden Layer 和 2x1 Output Layer 為例

Output Layer 和 Cost Function 的數學式為

$$ net_{o1} = w_1 * x_1 + w_2 * x_2 + b_{o1} $$

$$ out_{o1} = sigmoid(net_{o1}) $$

$$ E = \frac{1}{2}(y - out_{o1})^2 $$

對於 Output Layer w1 的偏微分連鎖律運算

$$ \frac{\delta E}{\delta w_1} = \frac{\delta E}{\delta out_{o1}} * \frac{\delta out_{o1}}{\delta net_{o1}} * \frac{\delta net_{o1}}{\delta w_1} $$

$$ \frac{\delta E}{\delta out_{o1}} = \frac{\delta \frac{1}{2}(y - out_{o1})^2}{\delta out_{o1}} = \frac{1}{2} * 2 * (y - out_{o1}) * -1 = out_{o1} - y $$

$$ \frac{\delta out_{o1}}{\delta net_{o1}} = out_{o1} * (1 - out_{o1}) $$

$$ \frac{\delta net_{o1}}{\delta w_1} = x_1 $$

Hidden Layer 的數學式

$$ net_{h1} = w_5 * x_5 + w_6 * x_6 + b_{h1} $$
$$ out_{h1} = sigmoid(net_{h1}) $$
$$ x_1 = out_{h1} $$

對於 Hidden Layer w5 的偏微分連鎖律運算

$$ \frac{\delta E}{\delta w_5} = (\frac{\delta E}{\delta out_{o1}} * \frac{\delta out_{o1}}{\delta net_{o1}} * \frac{\delta net_{o1}}{\delta out_{h1}} + \frac{\delta E}{\delta out_{o2}} * \frac{\delta out_{o2}}{\delta net_{o2}} * \frac{\delta net_{o2}}{\delta out_{h1}}) * \frac{\delta out_{h1}}{\delta net_{h1}} * \frac{\delta net_{h1}}{\delta w_5} $$

其中括號內的運算可以在前一個 Layer 進行 Back Propagation 時算完並傳下去

更新的方法是將 weight 減去 計算出的 delta 乘上 learning_rate 後並更新 weight

Python 實現為

```python
def backward(self, prev_delta, learning_rate):
    # prev_delta 為前一層算出的 delta (error)
    delta = prev_delta * d_sigmoid(self.out)
    next_delta = np.matmul(self.w, delta)

    self.w -= np.multiply.outer(self.x, delta) * learning_rate
    self.b -= delta * learning_rate

    if self.prev:
        self.prev.backward(next_delta, learning_rate)

```

## Result

所有測試均已加上

```python
np.random.seed(0)
```

確保可重製性

### 2x2 Hidden Layer and 2x1 Output Layer

#### Epochs 40000 learning rate 0.1

```
[0 0] [0.02138379]
[0 1] [0.98147129]
[1 0] [0.98143743]
[1 1] [0.01919473]
```

![](https://i.imgur.com/ZtcbHTJ.png)

#### Epochs 40000 learning rate 1

```
[0 0] [0.00628088]
[0 1] [0.99468687]
[1 0] [0.99468511]
[1 1] [0.00541511]
```

![](https://i.imgur.com/7JYIyoP.png)

### 2x2 Hidden Layer1, 2x2 Hidden Layer2 and 2x1 Output Layer

#### Epochs 40000 learning rate 0.1

```
[0 0] [0.01033454]
[0 1] [0.98957752]
[1 0] [0.98950115]
[1 1] [0.01087367]
```

![](https://i.imgur.com/58y63rI.png)

####  Epochs 40000 learning rate 1

```
[0 0] [0.00249808]
[0 1] [0.99593212]
[1 0] [0.99593971]
[1 1] [0.00631386]
```

![](https://i.imgur.com/xpfW9l4.png)

### 2x2 Hidden Layer1, 2x4 Hidden Layer2 and 4x1 Output Layer

#### Epochs 40000 learning rate 0.1

```
[0 0] [0.0088511]
[0 1] [0.99000213]
[1 0] [0.98995251]
[1 1] [0.00981338]
```

![](https://i.imgur.com/BmgUKtD.png)

#### Epochs 40000 learning rate 1

```
[0 0] [0.00260317]
[0 1] [0.99727434]
[1 0] [0.99727158]
[1 1] [0.00250345]
```

![](https://i.imgur.com/MHsPOuB.png)

## Discussion

### learning rate 和層數

從前面的結果來看，較大的 learning rate 的確可以加快訓練的速度

較多的層數也可以讓準確率提升

### PyTorch-like Interface

在這次 Lab 中，為了能夠快速地增加層數和改變層的大小，所以做了像 Pytorch 的介面

```
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
```

只要將 forward 裡定義好 Neural Network 的層數和大小，就可以訓練了