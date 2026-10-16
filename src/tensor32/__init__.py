from abc import abstractmethod
from collections import OrderedDict
from enum import Enum, auto
from typing import override

import numpy as np
import torch
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray

from tensor32.exception import ArgumentError, MathError


def save(model: OrderedDict[str, torch.Tensor], path: str) -> None:
    torch.save(model, path)


def load(path: str) -> OrderedDict[str, torch.Tensor]:
    return torch.load(path)


class Operation(Enum):
    ADD = auto()
    NEG = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    ABS = auto()
    LOG = auto()
    T = auto()
    MT = auto()
    MATMUL = auto()
    RELU = auto()
    TANH = auto()
    SIGMOID = auto()
    MEAN = auto()


# if broadcasting was done to satisfy an operation,
# the gradients have to be accumulated across relevant axis during the backward pass
def unbroadcast(
    grad: NDArray[np.float32], init_shape: tuple[int, ...]
) -> NDArray[np.float32]:
    if grad.shape == init_shape:
        return grad

    # sum across the outer most dimension for the number dimensions added
    for _ in range(len(grad.shape) - len(init_shape)):
        grad = grad.sum(axis=0)

    # in the inner dimensions, broadcasting only happened if the shape is 1
    for i, dim in enumerate(init_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# a data class for constants that do not require gradient tracking
class Container:
    __slots__ = ("data",)

    def __init__(self, data: ArrayLike) -> None:
        if not isinstance(data, ndarray):
            data = np.array(data, dtype=np.float32)
        if data.dtype != np.float32:
            raise ArgumentError(
                f"Container.__init__: only supporting 32 bit float types, found dtype {data.dtype}"
            )

        self.data = data

    def __add__(self, operand_2: float | Container) -> Container:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Container(data=self.data + operand_2.data)

    def __radd__(self, operand_1: float) -> Container:
        return self + operand_1

    def __neg__(self) -> Container:
        return Container(data=-self.data)

    def __sub__(self, operand_2: float | Container) -> Container:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Container(data=self.data - operand_2.data)

    def __rsub__(self, operand_1: float) -> Container:
        return Container(data=operand_1 - self.data)

    def __mul__(self, operand_2: float | Container) -> Container:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Container(data=self.data * operand_2.data)

    def __rmul__(self, operand_1: float) -> Container:
        return self * operand_1

    def __truediv__(self, divisor: float | Container) -> Container:
        if isinstance(divisor, (int, float)):
            divisor = Container(data=divisor)
        if np.any(divisor.data == 0):
            raise MathError("Container.__truediv__: 0 value(s) caught in divisor")

        return Container(data=self.data / divisor.data)

    def __rtruediv__(self, numerator: float) -> Container:
        if np.any(self.data == 0):
            raise MathError("Container.__rtruediv__: 0 value(s) caught in divisor")

        return Container(data=numerator / self.data)

    def __pow__(self, exponent: float | Container) -> Container:
        if isinstance(exponent, (int, float)):
            exponent = Container(data=exponent)

        return Container(data=self.data**exponent.data)

    def __rpow__(self, base: float) -> Container:
        return Container(data=base**self.data)

    def abs(self) -> Container:
        return Container(data=np.abs(self.data))

    def log(self) -> Container:
        if np.any(self.data <= 0):
            raise MathError("Container.log: -ve or 0 value(s) caught in argument")

        return Container(data=np.log(self.data))

    @property
    def T(self) -> Container:
        # using copy because .T only creates a view
        return Container(data=self.data.T.copy())

    @property
    def mT(self) -> Container:
        if len(self.data.shape) < 2:
            raise MathError(
                f"Container.mT: cannot do matrix transpose on a 1D array, found shape {self.data.shape}"
            )

        # using copy because .swapaxes only creates a view
        return Container(data=np.swapaxes(self.data, -1, -2).copy())

    def __matmul__(self, matrix_2: Container) -> Container:
        if len(self.data.shape) < 2 or len(matrix_2.data.shape) < 2:
            raise MathError(
                f"Container.__matmul__: cannot do matrix multiplication on 1D arrays, found shapes {self.data.shape} and {matrix_2.data.shape}"
            )

        return Container(data=self.data @ matrix_2.data)

    def __rmatmul__(self, matrix_1: Container) -> Container:
        if len(matrix_1.data.shape) < 2 or len(self.data.shape) < 2:
            raise MathError(
                f"Container.__rmatmul__: cannot do matrix multiplication on 1D arrays, found shapes {matrix_1.data.shape} and {self.data.shape}"
            )

        return Container(data=matrix_1.data @ self.data)

    def relu(self) -> Container:
        return Container(data=np.maximum(0, self.data))

    def tanh(self) -> Container:
        return Container(data=np.tanh(self.data))

    def sigmoid(self) -> Container:
        # numerically stable sigmoid, prevents division by inf and overflow
        return Container(
            data=np.where(
                self.data >= 0,
                1 / (1 + np.exp(-self.data)),
                np.exp(self.data) / (1 + np.exp(self.data)),
            )
        )

    def mean(self) -> Container:
        return Container(data=float(np.mean(self.data)))


# a class for numbers that do require gradient tracking
class Tensor(Container):
    __slots__ = ("grad", "operands", "operation")

    def __init__(
        self,
        data: ArrayLike,
        operands: tuple[Container | Tensor, ...] | None = None,
        operation: Operation | None = None,
    ) -> None:
        super().__init__(data)
        self.grad = np.zeros_like(data)
        self.operands = operands
        self.operation = operation

    @override
    def __add__(self, operand_2: float | Container | Tensor) -> Tensor:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Tensor(
            data=self.data + operand_2.data,
            operands=(self, operand_2),
            operation=Operation.ADD,
        )

    @override
    def __radd__(self, operand_1: float | Container) -> Tensor:
        return self + operand_1

    @override
    def __neg__(self) -> Tensor:
        return Tensor(data=-self.data, operands=(self,), operation=Operation.NEG)

    @override
    def __sub__(self, operand_2: float | Container | Tensor) -> Tensor:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Tensor(
            data=self.data - operand_2.data,
            # order of operands only matter in -, /, ** and @
            operands=(self, operand_2),
            operation=Operation.SUB,
        )

    @override
    def __rsub__(self, operand_1: float | Container) -> Tensor:
        if isinstance(operand_1, (int, float)):
            operand_1 = Container(data=operand_1)

        return Tensor(
            data=operand_1.data - self.data,
            operands=(operand_1, self),
            operation=Operation.SUB,
        )

    @override
    def __mul__(self, operand_2: float | Container | Tensor) -> Tensor:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Tensor(
            data=self.data * operand_2.data,
            operands=(self, operand_2),
            operation=Operation.MUL,
        )

    @override
    def __rmul__(self, operand_1: float | Container) -> Tensor:
        return self * operand_1

    @override
    def __truediv__(self, divisor: float | Container | Tensor) -> Tensor:
        if isinstance(divisor, (int, float)):
            divisor = Container(data=divisor)
        if np.any(divisor.data == 0):
            raise MathError("Tensor.__truediv__: 0 value(s) caught in divisor")

        return Tensor(
            data=self.data / divisor.data,
            operands=(self, divisor),
            operation=Operation.DIV,
        )

    @override
    def __rtruediv__(self, numerator: float | Container) -> Tensor:
        if np.any(self.data == 0):
            raise MathError("Tensor.__rtruediv__: 0 value(s) caught in divisor")
        if isinstance(numerator, (int, float)):
            numerator = Container(data=numerator)

        return Tensor(
            data=numerator.data / self.data,
            operands=(numerator, self),
            operation=Operation.DIV,
        )

    @override
    def __pow__(self, exponent: float | Container | Tensor) -> Tensor:
        if isinstance(exponent, (int, float)):
            exponent = Container(data=exponent)

        return Tensor(
            data=self.data**exponent.data,
            operands=(self, exponent),
            operation=Operation.POW,
        )

    @override
    def __rpow__(self, base: float | Container) -> Tensor:
        if isinstance(base, (int, float)):
            base = Container(data=base)

        return Tensor(
            data=base.data**self.data,
            operands=(base, self),
            operation=Operation.POW,
        )

    @override
    def abs(self) -> Tensor:
        return Tensor(
            data=np.abs(self.data),
            operands=(self,),
            operation=Operation.ABS,
        )

    @override
    def log(self) -> Tensor:
        if np.any(self.data <= 0):
            raise MathError("Tensor.log: -ve or 0 value(s) caught in argument")

        return Tensor(
            data=np.log(self.data),
            operands=(self,),
            operation=Operation.LOG,
        )

    @override
    @property
    def T(self) -> Tensor:
        return Tensor(
            data=self.data.T.copy(),
            operands=(self,),
            operation=Operation.T,
        )

    @override
    @property
    def mT(self) -> Tensor:
        if len(self.data.shape) < 2:
            raise MathError(
                f"Tensor.mT: cannot do matrix transpose on a 1D tensor, found shape {self.data.shape}"
            )

        return Tensor(
            data=np.swapaxes(self.data, -1, -2).copy(),
            operands=(self,),
            operation=Operation.MT,
        )

    @override
    def __matmul__(self, matrix_2: Container | Tensor) -> Tensor:
        if len(self.data.shape) < 2 or len(matrix_2.data.shape) < 2:
            raise MathError(
                f"Tensor.__matmul__: cannot do matrix multiplication on 1D tensors, found shapes {self.data.shape} and {matrix_2.data.shape}"
            )

        return Tensor(
            data=self.data @ matrix_2.data,
            operands=(self, matrix_2),
            operation=Operation.MATMUL,
        )

    @override
    def __rmatmul__(self, matrix_1: Container) -> Tensor:
        if len(matrix_1.data.shape) < 2 or len(self.data.shape) < 2:
            raise MathError(
                f"Tensor.__rmatmul__: cannot do matrix multiplication on 1D tensors, found shapes {matrix_1.data.shape} and {self.data.shape}"
            )

        return Tensor(
            data=matrix_1.data @ self.data,
            operands=(matrix_1, self),
            operation=Operation.MATMUL,
        )

    @override
    def relu(self) -> Tensor:
        return Tensor(
            data=np.maximum(0, self.data),
            operands=(self,),
            operation=Operation.RELU,
        )

    @override
    def tanh(self) -> Tensor:
        return Tensor(
            data=np.tanh(self.data),
            operands=(self,),
            operation=Operation.TANH,
        )

    @override
    def sigmoid(self) -> Tensor:
        return Tensor(
            data=np.where(
                self.data >= 0,
                1 / (1 + np.exp(-self.data)),
                np.exp(self.data) / (1 + np.exp(self.data)),
            ),
            operands=(self,),
            operation=Operation.SIGMOID,
        )

    @override
    def mean(self) -> Tensor:
        return Tensor(
            data=np.mean(self.data), operands=(self,), operation=Operation.MEAN
        )

    # non-recursive topological sort
    def __topo_sort(self) -> list[Container | Tensor]:
        ordered = []
        visited = set()
        stack = [self]

        while stack:
            top = stack[-1]
            if top not in visited:
                visited.add(top)
                if isinstance(top, Tensor) and top.operands:
                    for operand in top.operands:
                        stack.append(operand)  # type: ignore
            else:
                top = stack.pop()
                if top not in ordered:
                    ordered.append(top)

        return ordered

    def backward(self) -> None:
        ordered = self.__topo_sort()
        self.grad = np.ones_like(self.data)
        for elem in reversed(ordered):
            if not isinstance(elem, Tensor) or not elem.operation:
                continue
            assert elem.operands  # assertions for lsp only

            # chain rule
            match elem.operation:
                case Operation.ADD:
                    op_1, op_2 = elem.operands
                    if isinstance(op_1, Tensor):
                        op_1.grad += unbroadcast(elem.grad, op_1.grad.shape)
                    if isinstance(op_2, Tensor):
                        op_2.grad += unbroadcast(elem.grad, op_2.grad.shape)
                case Operation.NEG:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad -= elem.grad
                case Operation.SUB:
                    op_1, op_2 = elem.operands
                    if isinstance(op_1, Tensor):
                        op_1.grad += unbroadcast(elem.grad, op_1.grad.shape)
                    if isinstance(op_2, Tensor):
                        op_2.grad += unbroadcast(-elem.grad, op_2.grad.shape)
                case Operation.MUL:
                    op_1, op_2 = elem.operands
                    if isinstance(op_1, Tensor):
                        op_1.grad += unbroadcast(elem.grad * op_2.data, op_1.grad.shape)
                    if isinstance(op_2, Tensor):
                        op_2.grad += unbroadcast(elem.grad * op_1.data, op_2.grad.shape)
                case Operation.DIV:
                    op_1, op_2 = elem.operands
                    if isinstance(op_1, Tensor):
                        op_1.grad += unbroadcast(elem.grad / op_2.data, op_1.grad.shape)
                    if isinstance(op_2, Tensor):
                        op_2.grad += unbroadcast(
                            elem.grad * -op_1.data / op_2.data**2, op_2.grad.shape
                        )
                case Operation.POW:
                    op_1, op_2 = elem.operands
                    if isinstance(op_1, Tensor):
                        op_1.grad += unbroadcast(
                            elem.grad * op_2.data * op_1.data ** (op_2.data - 1),
                            op_1.grad.shape,
                        )
                    if isinstance(op_2, Tensor):
                        if np.any(op_1.data <= 0):
                            raise MathError(
                                "Tensor.backward: -ve or 0 value(s) caught in base"
                            )
                        op_2.grad += unbroadcast(
                            elem.grad * np.log(op_1.data) * op_1.data**op_2.data,
                            op_2.grad.shape,
                        )
                case Operation.ABS:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    # the derivative of abs(x) does not exist at x = 0, so assigning the derivative then to 0
                    op.grad += elem.grad * np.sign(op.data)
                case Operation.LOG:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad * 1 / op.data
                case Operation.T:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad.T
                case Operation.MT:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += np.swapaxes(elem.grad, -1, -2)
                case Operation.MATMUL:
                    op_1, op_2 = elem.operands
                    # using swapaxes prevents messing up batch dimensions if any
                    if isinstance(op_1, Tensor):
                        op_1.grad += unbroadcast(
                            elem.grad @ np.swapaxes(op_2.data, -1, -2), op_1.grad.shape
                        )
                    if isinstance(op_2, Tensor):
                        op_2.grad += unbroadcast(
                            np.swapaxes(op_1.data, -1, -2) @ elem.grad, op_2.grad.shape
                        )
                case Operation.RELU:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad * (op.data > 0).astype(np.float32)
                case Operation.TANH:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad * (1 - elem.data**2)
                case Operation.SIGMOID:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad * elem.data * (1 - elem.data)
                case Operation.MEAN:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad / op.data.size
