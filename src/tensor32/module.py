from abc import abstractmethod
from typing import override

import numpy as np

from tensor32 import Container, Tensor
from tensor32.config import Config


class Module:
    @abstractmethod
    def forward(self, X: Container | Tensor) -> Container | Tensor: ...

    @abstractmethod
    def __call__(self, X: Container | Tensor) -> Container | Tensor: ...


class Linear(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        rng = np.random.default_rng()

        # bound for Kaiming uniform initialization in fan in mode
        w_bound = np.sqrt(6 / in_channels)
        self.weights = Tensor(
            data=rng.uniform(
                low=-w_bound, high=w_bound, size=(out_channels, in_channels)
            ).astype(np.float32)
        )

        # bound for scaled LeCun uniform initialization
        b_bound = np.sqrt(1 / in_channels)
        self.biases = Tensor(
            data=rng.uniform(low=-b_bound, high=b_bound, size=out_channels).astype(
                np.float32
            )
        )

    @override
    def forward(self, X: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            return X @ self.weights.T + self.biases
        else:
            return Container(data=X.data @ self.weights.data.T + self.biases.data)

    @override
    def __call__(self, X: Container | Tensor) -> Container | Tensor:
        return self.forward(X)


class ReLU(Module):
    @override
    def forward(self, X: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            assert isinstance(X, Tensor)
            return X.relu()
        else:
            return Container(data=np.maximum(0, X.data, dtype=np.float32))

    @override
    def __call__(self, X: Container | Tensor) -> Container | Tensor:
        return self.forward(X)


class Tanh(Module):
    @override
    def forward(self, X: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            assert isinstance(X, Tensor)
            return X.tanh()
        else:
            return Container(data=np.tanh(X.data, dtype=np.float32))

    @override
    def __call__(self, X: Container | Tensor) -> Container | Tensor:
        return self.forward(X)


class Sigmoid(Module):
    @override
    def forward(self, X: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            assert isinstance(X, Tensor)
            return X.sigmoid()
        else:
            # numerically stable sigmoid, prevents division by inf and overflow
            return Container(
                data=np.where(
                    X.data >= 0,
                    1 / (1 + np.exp(-X.data)),
                    np.exp(X.data) / (1 + np.exp(X.data)),
                ).astype(np.float32)
            )

    @override
    def __call__(self, X: Container | Tensor) -> Container | Tensor:
        return self.forward(X)
