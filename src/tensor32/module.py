from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Literal, overload, override

import numpy as np
import torch

from tensor32 import Container, Tensor
from tensor32.exception import ArgumentError


class Module:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Container | Tensor: ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Container | Tensor: ...


class Stateless: ...


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        rng = np.random.default_rng()

        # bound for Kaiming uniform initialization in fan in mode
        w_bound = np.sqrt(6 / in_features)
        self.weight = Tensor(
            data=rng.uniform(
                low=-w_bound, high=w_bound, size=(out_features, in_features)
            ).astype(np.float32)
        )

        # bound for scaled LeCun uniform initialization
        b_bound = np.sqrt(1 / in_features)
        self.bias = Tensor(
            data=rng.uniform(low=-b_bound, high=b_bound, size=out_features).astype(
                np.float32
            )
        )

    @overload
    def forward(self, X: Container | Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def forward(self, X: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def forward(self, X: Container | Tensor, grad: bool = True) -> Container | Tensor:
        if len(X.data.shape) < 2:
            raise ArgumentError(
                f"Linear.forward: expected a multi-dimensional input with batch size(s) first, found shape {X.data.shape}"
            )
        if grad:
            return X @ self.weight.T + self.bias
        else:
            return Container(data=X.data @ self.weight.data.T + self.bias.data)

    @overload
    def __call__(self, X: Container | Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, X: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, X: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(X, grad)


class ReLU(Module, Stateless):
    @overload
    def forward(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def forward(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def forward(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        if grad:
            if not isinstance(z, Tensor):
                raise ArgumentError(
                    f"ReLU.forward: expected a tensor under gradient tracking context, found {type(z).__name__}"
                )
            return z.relu()
        else:
            return Container(data=np.maximum(0, z.data, dtype=np.float32))

    @overload
    def __call__(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(z, grad)  # type: ignore


class Tanh(Module, Stateless):
    @overload
    def forward(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def forward(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def forward(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        if grad:
            if not isinstance(z, Tensor):
                raise ArgumentError(
                    f"Tanh.forward: expected a tensor under gradient tracking context, found {type(z).__name__}"
                )
            return z.tanh()
        else:
            return Container(data=np.tanh(z.data, dtype=np.float32))

    @overload
    def __call__(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(z, grad)  # type: ignore


class Sigmoid(Module, Stateless):
    @overload
    def forward(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def forward(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def forward(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        if grad:
            if not isinstance(z, Tensor):
                raise ArgumentError(
                    f"Sigmoid.forward: expected a tensor under gradient tracking context, found {type(z).__name__}"
                )
            return z.sigmoid()
        else:
            # numerically stable sigmoid, prevents division by inf and overflow
            return Container(
                data=np.where(
                    z.data >= 0,
                    1 / (1 + np.exp(-z.data)),
                    np.exp(z.data) / (1 + np.exp(z.data)),
                ).astype(np.float32)
            )

    @overload
    def __call__(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(z, grad)  # type: ignore


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        for module in args:
            if isinstance(module, Sequential):
                raise ArgumentError(
                    "Sequential.__init__: cannot nest sequential classes"
                )
        self.modules = args

    @overload
    def forward(self, X: Container | Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def forward(self, X: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def forward(self, X: Container | Tensor, grad: bool = True) -> Container | Tensor:
        state = X
        for module in self.modules:
            state = module(state, grad)

        return state

    @overload
    def __call__(self, X: Container | Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, X: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, X: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(X, grad)

    def parameters(self) -> list[Tensor]:
        params = []
        for module in self.modules:
            if issubclass(type(module), Stateless):
                continue
            for val in vars(module).values():
                if isinstance(val, Tensor):
                    params.append(val)

        return params

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict()
        for i, module in enumerate(self.modules):
            if issubclass(type(module), Stateless):
                continue
            for key, val in vars(module).items():
                if isinstance(val, Tensor):
                    state[f"{i}.{key}"] = torch.from_numpy(val.data)

        return state


class MSELoss(Module, Stateless):
    @overload
    def forward(
        self, yc: Container | Tensor, y: Container | Tensor, grad: Literal[True] = True
    ) -> Tensor: ...
    @overload
    def forward(
        self, yc: Container | Tensor, y: Container | Tensor, grad: Literal[False]
    ) -> Container: ...
    @override
    def forward(
        self, yc: Container | Tensor, y: Container | Tensor, grad: bool = True
    ) -> Container | Tensor:
        if grad:
            if not isinstance(yc, Tensor) or not isinstance(y, Tensor):
                raise ArgumentError(
                    f"MSELoss.forward: expected a tensor under gradient tracking context, found {type(yc).__name__} and {type(y).__name__}"
                )
            return ((yc - y) ** 2).mean()
        else:
            return Container(data=float(((yc.data - y.data) ** 2).mean()))

    @overload
    def __call__(
        self, yc: Container | Tensor, y: Container | Tensor, grad: Literal[True] = True
    ) -> Tensor: ...
    @overload
    def __call__(
        self, yc: Container | Tensor, y: Container | Tensor, grad: Literal[False]
    ) -> Container: ...
    @override
    def __call__(
        self, yc: Container | Tensor, y: Container | Tensor, grad: bool = True
    ) -> Container | Tensor:
        return self.forward(yc, y, grad)
