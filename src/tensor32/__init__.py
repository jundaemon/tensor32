from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class Operation(Enum):
    ADD = auto()
    NEG = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    T = auto()
    MT = auto()
    MATMUL = auto()


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

    def __init__(self, data: float | NDArray[np.float32]) -> None:
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)

        self.data = data


# a class for numbers that do require gradient tracking
class Tensor(Container):
    __slots__ = ("grad", "operands", "operation")

    def __init__(
        self,
        data: float | NDArray[np.float32],
        operands: tuple[Container | Tensor, ...] | None = None,
        operation: Operation | None = None,
    ) -> None:
        super().__init__(data)
        self.grad = np.zeros_like(data, dtype=np.float32)
        self.operands = operands
        self.operation = operation

    def __add__(self, operand_2: float | Container | Tensor) -> Tensor:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Tensor(
            data=self.data + operand_2.data,
            operands=(self, operand_2),
            operation=Operation.ADD,
        )

    def __radd__(self, operand_1: float | Container) -> Tensor:
        return self + operand_1

    def __neg__(self) -> Tensor:
        return Tensor(data=-self.data, operands=(self,), operation=Operation.NEG)

    def __sub__(self, operand_2: float | Container | Tensor) -> Tensor:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Tensor(
            data=self.data - operand_2.data,
            # order of operands only matter in -, /, ** and @
            operands=(self, operand_2),
            operation=Operation.SUB,
        )

    def __rsub__(self, operand_1: float | Container) -> Tensor:
        if isinstance(operand_1, (int, float)):
            operand_1 = Container(data=operand_1)

        return Tensor(
            data=operand_1.data - self.data,
            operands=(operand_1, self),
            operation=Operation.SUB,
        )

    def __mul__(self, operand_2: float | Container | Tensor) -> Tensor:
        if isinstance(operand_2, (int, float)):
            operand_2 = Container(data=operand_2)

        return Tensor(
            data=self.data * operand_2.data,
            operands=(self, operand_2),
            operation=Operation.MUL,
        )

    def __rmul__(self, operand_1: float | Container) -> Tensor:
        return self * operand_1

    def __truediv__(self, divisor: float | Container | Tensor) -> Tensor:
        if isinstance(divisor, (int, float)):
            divisor = Container(data=divisor)
        assert np.all(divisor.data != 0)

        return Tensor(
            data=self.data / divisor.data,
            operands=(self, divisor),
            operation=Operation.DIV,
        )

    def __rtruediv__(self, numerator: float | Container) -> Tensor:
        assert np.all(self.data != 0)
        if isinstance(numerator, (int, float)):
            numerator = Container(data=numerator)

        return Tensor(
            data=numerator.data / self.data,
            operands=(numerator, self),
            operation=Operation.DIV,
        )

    # no __rpow__ because I think it doesn't make sense
    def __pow__(self, exponent: float | Container | Tensor) -> Tensor:
        if isinstance(exponent, (int, float)):
            exponent = Container(data=exponent)

        return Tensor(
            data=self.data**exponent.data,
            operands=(self, exponent),
            operation=Operation.POW,
        )

    def T(self) -> Tensor:
        assert len(self.data.shape) >= 2
        return Tensor(
            # using copy because .T only creates a view
            data=self.data.T.copy(),
            operands=(self,),
            operation=Operation.T,
        )

    def mT(self) -> Tensor:
        assert len(self.data.shape) >= 2
        return Tensor(
            # using copy because swapaxes only creates a view
            data=np.swapaxes(self.data, -1, -2).copy(),
            operands=(self,),
            operation=Operation.MT,
        )

    def __matmul__(self, matrix_2: Container | Tensor) -> Tensor:
        assert len(self.data.shape) >= 2
        assert len(matrix_2.data.shape) >= 2

        return Tensor(
            data=self.data @ matrix_2.data,
            operands=(self, matrix_2),
            operation=Operation.MATMUL,
        )

    def __rmatmul__(self, matrix_1: Container) -> Tensor:
        assert len(matrix_1.data.shape) >= 2
        assert len(self.data.shape) >= 2

        return Tensor(
            data=matrix_1.data @ self.data,
            operands=(matrix_1, self),
            operation=Operation.MATMUL,
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
        self.grad = np.ones_like(self.data, dtype=np.float32)
        for elem in reversed(ordered):
            if not isinstance(elem, Tensor) or not elem.operation:
                continue
            assert elem.operands

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
                    op.grad += -elem.grad
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
                    # will rarely or never find the gradient of the exponent
                    if isinstance(op_2, Tensor):
                        assert np.all(op_1.data > 0)
                        op_2.grad += unbroadcast(
                            elem.grad * np.log(op_1.data) * op_1.data**op_2.data,
                            op_2.grad.shape,
                        )
                case Operation.T:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += elem.grad.T
                case Operation.MT:
                    assert isinstance((op := elem.operands[0]), Tensor)
                    op.grad += np.swapaxes(elem.grad, -1, -2)
                case Operation.MATMUL:
                    op_1, op_2 = elem.operands
                    if isinstance(op_1, Tensor):
                        # using swapaxes prevents messing up batch dimensions if any
                        op_1.grad += unbroadcast(
                            elem.grad @ np.swapaxes(op_2.data, -1, -2), op_1.grad.shape
                        )
                    if isinstance(op_2, Tensor):
                        op_2.grad += unbroadcast(
                            np.swapaxes(op_1.data, -1, -2) @ elem.grad, op_2.grad.shape
                        )
