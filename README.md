## Introduction
tensor32 is an autograd engine built on top of floating 32 bit numpy arrays, written to learn forward passes, backpropagation, loss functions and model optimization. It comes with features like:
- Pytorch like API
- Linear layers
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (MSE, BCEWithLogits)
- Optimizers (SGD, SGDwM, RMSprop, Adam)

## Disclaimer
This is a hobby project and should not be used in production.

## Examples
Here is an [example](https://github.com/jundaemon/tensor32/blob/main/src/example_training.py) of training a small Multi Layer Perceptron (MLP) to solve the XOR problem.
```python
from tensor32 import Container, save
from tensor32.module import BCEWithLogitsLoss, Linear, ReLU, Sequential
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
        Linear(2, 2),
        ReLU(),
        Linear(2, 1),
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
from tensor32.module import Linear, ReLU, Sequential

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
        Linear(2, 2),
        ReLU(),
        Linear(2, 1),
    )
    model.load_state_dict(load("assets/model.pth"))

    for feature, label in zip(X, y):
        pred = model(feature, grad=False).sigmoid()
        print(f"actual: {label.data}, predicted: {pred.data}")
```

## Architecture
### Containers and Tensors
The core of this autograd engine involves operations between containers and tensors. Containers are wrappers around 32 bit floating numpy arrays while tensors inherit from containers and carry additional information like the operand(s) and operation used in its creation as well as the gradient of each floating point number. This ensures that constants do not carry additional overhead in memory. An alternative to such an architecture involves just having tensors and saving the operands, operation and gradients as `None` if gradient tracking is not needed. Having this alternative architecture leads to each tensor carrying an additional 24 bytes of memory just to have references to the `None` singleton.

As a result, the operation between a float/container and a tensor will always result in a tensor and an operation on just a tensor will lead to a tensor. The choice of requiring gradient tracking is shifted up to the module level. If gradient tracking is needed, modules will make use of dunder methods in tensors, if gradient tracking is not needed, then modules will make use of dunder methods in containers.

The backward pass in tensors is quite straight forward. The computational graph is topologically sorted using a non-recursive depth first traversal on the operands, the sort is then reversed and the gradient of the output of the graph is assigned as one, finally the sort is iterated through and chain rule is used to calculate the derivative of final output with respect to each tensor based on the operation used to create each tensor.

### Modules
Moving on to modules, they largely follow the implementation in Pytorch, except instead of a context manager like `no_grad()` in Pytorch, the choice of tracking gradients and building a computational graph is expressed through a boolean input argument to the forward pass of modules. This was a direct result of the container/tensor architecture. Type hints do not have access to the state of context managers, if I were to implement a context manager and type hint the return value of the `forward` method to be `Container | Tensor`, the Language Server Protocol (LSP) of the user would flag `backward` to not be implemented in the container class even though for certain, the output of `forward` would be a tensor under a gradient tracking context. Thus, using a boolean input argument to allow users to choose whether they want gradient tracking allows me to overload the `forward` and `__call__` methods and provide more precise type hints for each scenario.

### Optimizers
Finally for the optimizers, they also largely follow the implementation in Pytorch, except variants of the same optimizer utilize the factory pattern to save memory. For example in Stochastic Gradient Descent (SGD), if a momentum is given, it will instead initialize a SGD with Momentum (SGDwM) optimizer which carries an extra momentum term and array of velocities. This ensures that if momentum is not needed, then the SGD optimizer doesn't hold 8 byte references to the `None` singleton in the `momentum` and `velocities` properties.
