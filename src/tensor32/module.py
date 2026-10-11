from abc import abstractmethod
from collections import OrderedDict
from typing import Any, override

import numpy as np
import torch

from tensor32 import Container, Tensor
from tensor32.config import Config
from tensor32.exception import MissingTensorError


class Module:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Container | Tensor: ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Container | Tensor: ...


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
    def forward(self, input: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            return input @ self.weight.T + self.bias
        else:
            return Container(data=input.data @ self.weight.data.T + self.bias.data)

    @override
    def __call__(self, input: Container | Tensor) -> Container | Tensor:
        return self.forward(input)


class ReLU(Module, Stateless):
    @override
    def forward(self, input: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            # without no_grad, input should be a tensor
            if not isinstance(input, Tensor):
                raise MissingTensorError(
                    f"ReLU.forward: expected a Tensor under gradient tracking context, found {type(input).__name__}"
                )
            return input.relu()
        else:
            return Container(data=np.maximum(0, input.data, dtype=np.float32))

    @override
    def __call__(self, input: Container | Tensor) -> Container | Tensor:
        return self.forward(input)


class Tanh(Module, Stateless):
    @override
    def forward(self, input: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            if not isinstance(input, Tensor):
                raise MissingTensorError(
                    f"Tanh.forward: expected a Tensor under gradient tracking context, found {type(input).__name__}"
                )
            return input.tanh()
        else:
            return Container(data=np.tanh(input.data, dtype=np.float32))

    @override
    def __call__(self, input: Container | Tensor) -> Container | Tensor:
        return self.forward(input)


class Sigmoid(Module, Stateless):
    @override
    def forward(self, input: Container | Tensor) -> Container | Tensor:
        if Config.grad_enabled:
            if not isinstance(input, Tensor):
                raise MissingTensorError(
                    f"Sigmoid.forward: expected a Tensor under gradient tracking context, found {type(input).__name__}"
                )
            return input.sigmoid()
        else:
            # numerically stable sigmoid, prevents division by inf and overflow
            return Container(
                data=np.where(
                    input.data >= 0,
                    1 / (1 + np.exp(-input.data)),
                    np.exp(input.data) / (1 + np.exp(input.data)),
                ).astype(np.float32)
            )

    @override
    def __call__(self, input: Container | Tensor) -> Container | Tensor:
        return self.forward(input)


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        self.modules = args

    @override
    def forward(self, input: Container | Tensor) -> Container | Tensor:
        state = input
        for module in self.modules:
            state = module(state)

        return state

    @override
    def __call__(self, input: Container | Tensor) -> Container | Tensor:
        return self.forward(input)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict()
        for i, module in enumerate(self.modules):
            if issubclass(type(module), Stateless):
                continue

            for key, val in vars(module).items():
                state[f"{i}.{key}"] = torch.from_numpy(val.data)

        return state


class MSELoss(Module, Stateless):
    @override
    def forward(
        self, pred: Container | Tensor, actual: Container | Tensor
    ) -> Container | Tensor:
        if Config.grad_enabled:
            # without no_grad, either pred or actual has to be a tensor
            if not isinstance(pred, Tensor) or not isinstance(actual, Tensor):
                raise MissingTensorError(
                    f"MSELoss.forward: expected a Tensor under gradient tracking context, found {type(pred).__name__} and {type(actual).__name__}"
                )

            return ((pred - actual) ** 2).mean()
        else:
            return Container(data=float(((pred.data - actual.data) ** 2).mean()))

    @override
    def __call__(
        self, pred: Container, actual: Container | Tensor
    ) -> Container | Tensor:
        return self.forward(pred, actual)
