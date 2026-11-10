## Introduction

tensor32 is an autograd engine built on top of floating 32 bit numpy arrays, written to learn forward passes, backpropagation, loss functions and model optimization. It comes with features like:

- Pytorch like API
- Linear layers
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (MSE, BCEWithLogits)
- Optimizers (SGD, SGDM, RMSprop, Adam)

## Disclaimer

This is a hobby project and should not be used in production.

## Examples

Here is an [example](https://github.com/jundaemon/tensor32/blob/main/src/example_training.py) of training a small Multi Layer Perceptron (MLP) to solve the XOR problem.

```python
from tensor32 import Container, save
from tensor32.module import BCEWithLogitsLoss, Linear, Sequential, Tanh
from tensor32.optimizer import Adam

if __name__ == "__main__":
    X = [
        Container([[0, 0]]),
        Container([[1, 0]]),
        Container([[0, 1]]),
        Container([[1, 1]]),
    ]
    y = [
        Container([[0]]),
        Container([[1]]),
        Container([[1]]),
        Container([[0]]),
    ]
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
    )
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())
    epochs = 30_000

    for epoch in range(epochs):
        for feature, label in zip(X, y):
            yc = model(feature)
            loss = loss_fn(yc, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    save(model.state_dict(), "assets/model.pth")
```

And here is an [example](https://github.com/jundaemon/tensor32/blob/main/src/example_inference.py) to run inference on the same model trained.

```python
from tensor32 import Container, load
from tensor32.module import Linear, Sequential, Tanh

if __name__ == "__main__":
    X = [
        Container([[0, 0]]),
        Container([[1, 0]]),
        Container([[0, 1]]),
        Container([[1, 1]]),
    ]
    y = [
        Container([[0]]),
        Container([[1]]),
        Container([[1]]),
        Container([[0]]),
    ]
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
    )
    model.load_state_dict(load("assets/model.pth"))

    for feature, label in zip(X, y):
        pred = model(feature, grad=False).sigmoid()
        print(f"actual: {label.data}, predicted: {pred.data}")
```
