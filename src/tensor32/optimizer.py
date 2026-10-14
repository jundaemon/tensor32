from abc import abstractmethod
from typing import Literal, overload, override

import numpy as np

from tensor32 import Tensor
from tensor32.exception import ArgumentError


class Optimizer:
    def __init__(self, parameters: list[Tensor]) -> None:
        self.parameters = parameters

    @abstractmethod
    def step(self) -> None: ...

    def zero_grad(self) -> None:
        for tensor in self.parameters:
            tensor.grad.fill(0)


class SGD(Optimizer):
    @overload
    def __new__(
        cls, parameters: list[Tensor], lr: float = 1e-3, momentum: Literal[0] = 0
    ) -> SGD: ...
    @overload
    def __new__(
        cls, parameters: list[Tensor], lr: float = 1e-3, momentum: float = 0
    ) -> SGDMomentum: ...
    def __new__(
        cls, parameters: list[Tensor], lr: float = 1e-3, momentum: float = 0
    ) -> SGD | SGDMomentum:
        if cls is SGD and momentum > 0:
            return super().__new__(SGDMomentum)

        return super().__new__(cls)

    def __init__(
        self, parameters: list[Tensor], lr: float = 1e-3, momentum: float = 0
    ) -> None:
        if lr <= 0:
            raise ArgumentError(
                f"SGD.__init__: learning rate should be more than 0, received {lr}"
            )
        if momentum < 0:
            raise ArgumentError(
                f"SGD.__init__: momentum should be more than or equals to 0, received {momentum}"
            )

        super().__init__(parameters)
        self.lr = lr

    @override
    def step(self) -> None:
        for tensor in self.parameters:
            tensor.data -= self.lr * tensor.grad


class SGDMomentum(SGD):
    def __init__(
        self, parameters: list[Tensor], lr: float = 1e-3, momentum: float = 0
    ) -> None:
        super().__init__(parameters, lr, momentum)
        self.momentum = momentum
        self.velocities = []
        for tensor in parameters:
            self.velocities.append(np.zeros_like(tensor.data, dtype=np.float32))

    @override
    def step(self) -> None:
        for tensor, velocity in zip(self.parameters, self.velocities):
            velocity *= self.momentum
            velocity += self.lr * tensor.grad
            tensor.data -= velocity
