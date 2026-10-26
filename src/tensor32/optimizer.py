from abc import abstractmethod
from typing import override

import numpy as np

from tensor32 import Tensor
from tensor32.exception import ArgumentError


class Optimizer:
    def __init__(self, params: list[Tensor]) -> None:
        self.params = params

    @abstractmethod
    def step(self) -> None: ...

    def zero_grad(self) -> None:
        for tensor in self.params:
            tensor.grad.fill(0)


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ArgumentError(
                f"SGD.__init__: learning rate should be more than 0, found {lr}"
            )

        super().__init__(params)
        self.lr = lr

    @override
    def step(self) -> None:
        for tensor in self.params:
            tensor.data -= self.lr * tensor.grad


class SGDM(Optimizer):
    def __init__(
        self, params: list[Tensor], lr: float = 1e-3, momentum: float = 0.9
    ) -> None:
        if lr <= 0:
            raise ArgumentError(
                f"SGD.__init__: learning rate should be more than 0, found {lr}"
            )
        if momentum <= 0:
            raise ArgumentError(
                f"SGDM.__init__: momentum should be more than 0, found {momentum}"
            )

        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = []
        for tensor in params:
            self.velocities.append(np.zeros_like(tensor.data))

    @override
    def step(self) -> None:
        for tensor, velocity in zip(self.params, self.velocities):
            velocity *= self.momentum
            velocity += self.lr * tensor.grad
            tensor.data -= velocity


class RMSprop(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ArgumentError(
                f"RMSprop.__init__: learning rate should be more than 0, found {lr}"
            )
        if alpha <= 0:
            raise ArgumentError(
                f"RMSprop.__init__: alpha should be more than 0, found {alpha}"
            )
        if eps <= 0:
            raise ArgumentError(
                f"RMSprop.__init__: eps should be more than 0, found {eps}"
            )

        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.moving_averages = []
        for tensor in params:
            self.moving_averages.append(np.zeros_like(tensor.data))

    @override
    def step(self) -> None:
        for tensor, moving_average in zip(self.params, self.moving_averages):
            moving_average *= self.alpha
            moving_average += (1 - self.alpha) * tensor.grad**2
            tensor.data -= self.lr * tensor.grad / (np.sqrt(moving_average) + self.eps)


class Adam(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ArgumentError(
                f"Adam.__init__: learning rate should be more than 0, found {lr}"
            )
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ArgumentError(
                f"Adam.__init__: betas should be between 0 (inclusive) and 1, found {betas[0]} and {betas[1]}"
            )
        if eps <= 0:
            raise ArgumentError(
                f"Adam.__init__: eps should be more than 0, found {eps}"
            )

        super().__init__(params)
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.eps = eps
        self.moment_estimates_1 = []
        self.moment_estimates_2 = []
        for tensor in params:
            self.moment_estimates_1.append(np.zeros_like(tensor.data))
            self.moment_estimates_2.append(np.zeros_like(tensor.data))
        self.t = 1

    @override
    def step(self) -> None:
        for tensor, moment_estimate_1, moment_estimate_2 in zip(
            self.params, self.moment_estimates_1, self.moment_estimates_2
        ):
            moment_estimate_1 *= self.beta_1
            moment_estimate_1 += (1 - self.beta_1) * tensor.grad
            moment_estimate_2 *= self.beta_2
            moment_estimate_2 += (1 - self.beta_2) * tensor.grad**2
            moment_estimate_1c = moment_estimate_1 / (1 - self.beta_1**self.t)
            moment_estimate_2c = moment_estimate_2 / (1 - self.beta_2**self.t)
            tensor.data -= (
                self.lr * moment_estimate_1c / (np.sqrt(moment_estimate_2c) + self.eps)
            )
        self.t += 1
