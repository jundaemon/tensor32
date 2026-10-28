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
            # the choice of having gradients for inputs is up to the user
            return X @ self.weight.T + self.bias
        else:
            # underlying numpy arrays are shared but not modified, there is little overhead in creating temporary containers
            if isinstance(X, Tensor):
                X = Container(data=X.data)
            temp_weight = Container(data=self.weight.data)
            temp_bias = Container(data=self.bias.data)
            return X @ temp_weight.T + temp_bias

    @overload
    def __call__(self, X: Container | Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, X: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, X: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(X, grad)


class ReLU(Module):
    @overload
    def forward(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def forward(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def forward(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        if grad:
            if not isinstance(z, Tensor):
                # unlike linear, if grad is true, input has to be a tensor
                raise ArgumentError(
                    f"ReLU.forward: expected a tensor under gradient tracking context, found {type(z).__name__}"
                )
            return z.relu()
        else:
            if isinstance(z, Tensor):
                z = Container(data=z.data)
            return z.relu()

    @overload
    def __call__(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(z, grad)  # type: ignore


class Tanh(Module):
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
            if isinstance(z, Tensor):
                z = Container(data=z.data)
            return z.tanh()

    @overload
    def __call__(self, z: Tensor, grad: Literal[True] = True) -> Tensor: ...
    @overload
    def __call__(self, z: Container | Tensor, grad: Literal[False]) -> Container: ...
    @override
    def __call__(self, z: Container | Tensor, grad: bool = True) -> Container | Tensor:
        return self.forward(z, grad)  # type: ignore


class Sigmoid(Module):
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
            if isinstance(z, Tensor):
                z = Container(data=z.data)
            return z.sigmoid()

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
            if isinstance(module, Loss):
                raise ArgumentError(
                    "Sequential.__init__: cannot accept loss functions in sequence"
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
            for val in vars(module).values():
                if isinstance(val, Tensor):
                    params.append(val)

        return params

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict()
        for i, module in enumerate(self.modules):
            for key, val in vars(module).items():
                if isinstance(val, (Container, Tensor)):
                    state[f"{i}.{key}"] = torch.from_numpy(val.data)

        return state

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        tensors = self.parameters()
        if len(tensors) != len(state_dict):
            raise ArgumentError(
                f"Sequential.load_state_dict: initialized state is incompatible with given state, expected {len(tensors)} tensors, found {len(state_dict)}"
            )

        for initialized, given in zip(tensors, state_dict.values()):
            if initialized.data.shape != given.shape:
                raise ArgumentError(
                    f"Sequential.load_state_dict: initialized tensor is incompatible with given tensor, expected shape {initialized.data.shape}, found {given.shape}"
                )
            if given.dtype != torch.float32:
                raise ArgumentError(
                    f"Sequential.load_state_dict: only supporting 32 bit float types, found {given.dtype}"
                )
            initialized.data = given.numpy()


class Loss(Module): ...


class MSELoss(Loss):
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
            # if grad is true, either input has to be tensor
            if not isinstance(yc, Tensor) and not isinstance(y, Tensor):
                raise ArgumentError(
                    f"MSELoss.forward: expected a tensor under gradient tracking context, found {type(yc).__name__} and {type(y).__name__}"
                )
            return ((yc - y) ** 2).mean()
        else:
            if isinstance(yc, Tensor):
                yc = Container(data=yc.data)
            if isinstance(y, Tensor):
                y = Container(data=y.data)
            return ((yc - y) ** 2).mean()

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


class BCEWithLogitsLoss(Loss):
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
            if not isinstance(yc, Tensor) and not isinstance(y, Tensor):
                raise ArgumentError(
                    f"BCEWithLogitsLoss.forward: expected a tensor under gradient tracking context, found {type(yc).__name__} and {type(y).__name__}"
                )
            return (yc.relu() - yc * y + (1 + np.e ** -yc.abs()).log()).mean()
        else:
            if isinstance(yc, Tensor):
                yc = Container(data=yc.data)
            if isinstance(y, Tensor):
                y = Container(data=y.data)
            return (yc.relu() - yc * y + (1 + np.e ** -yc.abs()).log()).mean()

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
