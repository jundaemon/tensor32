from abc import abstractmethod
from collections import OrderedDict
from typing import override

import numpy as np
import torch

from tensor32 import Container, Tensor
from tensor32.config import Config


class Module:
    @abstractmethod
    def forward(self, X: Container | Tensor) -> Container | Tensor: ...

    @abstractmethod
    def __call__(self, X: Container | Tensor) -> Container | Tensor: ...


class Stateless: ...


class Linear(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        rng = np.random.default_rng()

        # bound for Kaiming uniform initialization in fan in mode
        w_bound = np.sqrt(6 / in_channels)
        self.weight = Tensor(
            data=rng.uniform(
                low=-w_bound, high=w_bound, size=(out_channels, in_channels)
            ).astype(np.float32)
        )

        # bound for scaled LeCun uniform initialization
        b_bound = np.sqrt(1 / in_channels)
        self.bias = Tensor(
            data=rng.uniform(low=-b_bound, high=b_bound, size=out_channels).astype(
                np.float32
            )
        )

    @override
    def forward(self, X: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            return X @ self.weight.T + self.bias
        else:
            return Container(data=X.data @ self.weight.data.T + self.bias.data)

    @override
    def __call__(self, X: Container | Tensor) -> Container | Tensor:
        return self.forward(X)


class ReLU(Module, Stateless):
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


class Tanh(Module, Stateless):
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


class Sigmoid(Module, Stateless):
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


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        self.modules = args

    @override
    def forward(self, X: Container | Tensor) -> Container | Tensor:
        state = X
        for module in self.modules:
            state = module(state)

        return state

    @override
    def __call__(self, X: Container | Tensor) -> Container | Tensor:
        return self.forward(X)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict()
        for i, module in enumerate(self.modules):
            if isinstance(module, Stateless):
                continue

            for key, val in vars(module).items():
                state[f"{i}.{key}"] = torch.from_numpy(val.data)

        return state
