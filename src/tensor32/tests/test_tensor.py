import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from tensor32 import Tensor

RTOL = 1e-4
ATOL = 1e-4

OPS_SHAPES = [
    ((3, 3), (3, 3)),
    ((3, 3), (3, 1)),
    ((1, 4), (3, 4)),
    ((2, 3, 4), (1, 4)),
    ((2, 3, 4), (3, 1)),
]
MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 1, 3, 4), (2, 3, 4, 5)),
]


def helper_verify_binary_op(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    op_fn,  # avoid type hinting this to not spite lsp
    ensure_positive_a: bool = False,
    ensure_nonzero_b: bool = False,
) -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal(shape_a, dtype=np.float32)
    np_b = rng.standard_normal(shape_b, dtype=np.float32)

    if ensure_positive_a:
        np_a = np.abs(np_a) + 0.1  # avoid negative bases for powers
    if ensure_nonzero_b:
        np_b = np.sign(np_b) * np.maximum(np.abs(np_b), 0.1)  # avoid division by zero

    t_a = Tensor(np_a)
    t_b = Tensor(np_b)
    t_out = op_fn(t_a, t_b)
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_b = torch.tensor(np_b, requires_grad=True)
    pt_out = op_fn(pt_a, pt_b)
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(
        t_out.data,
        pt_out.detach().numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore
    assert_allclose(t_b.grad, pt_b.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


@pytest.mark.parametrize("shape_a, shape_b", OPS_SHAPES)
def test_add(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    helper_verify_binary_op(shape_a, shape_b, lambda a, b: a + b)


@pytest.mark.parametrize("shape_a, shape_b", OPS_SHAPES)
def test_sub(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    helper_verify_binary_op(shape_a, shape_b, lambda a, b: a - b)


@pytest.mark.parametrize("shape_a, shape_b", OPS_SHAPES)
def test_mul(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    helper_verify_binary_op(shape_a, shape_b, lambda a, b: a * b)


@pytest.mark.parametrize("shape_a, shape_b", OPS_SHAPES)
def test_div(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    helper_verify_binary_op(shape_a, shape_b, lambda a, b: a / b, ensure_nonzero_b=True)


@pytest.mark.parametrize("shape_a, shape_b", OPS_SHAPES)
def test_pow(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    helper_verify_binary_op(shape_a, shape_b, lambda a, b: a**b, ensure_positive_a=True)


@pytest.mark.parametrize("shape_a, shape_b", MATMUL_SHAPES)
def test_matmul(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    helper_verify_binary_op(shape_a, shape_b, lambda a, b: a @ b)


def test_neg() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((3, 3), dtype=np.float32)

    t_a = Tensor(np_a)
    t_out = -t_a
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = -pt_a
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


def test_T() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((3, 4), dtype=np.float32)

    t_a = Tensor(np_a)
    t_out = t_a.T
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = pt_a.T
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


def test_mT() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)

    t_a = Tensor(np_a)
    t_out = t_a.mT
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = pt_a.mT
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


def test_scalars() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((3, 3), dtype=np.float32)
    scalar = 10.3

    t_a = Tensor(np_a)
    t_out = (t_a * scalar + scalar) / scalar
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = (pt_a * scalar + scalar) / scalar
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


def test_relu() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((3, 3), dtype=np.float32)

    t_a = Tensor(np_a)
    t_out = t_a.relu()
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = pt_a.relu()
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


def test_tanh() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((3, 3), dtype=np.float32)

    t_a = Tensor(np_a)
    t_out = t_a.tanh()
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = pt_a.tanh()
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore


def test_sigmoid() -> None:
    rng = np.random.default_rng()
    np_a = rng.standard_normal((3, 3), dtype=np.float32)

    t_a = Tensor(np_a)
    t_out = t_a.sigmoid()
    t_out.backward()

    pt_a = torch.tensor(np_a, requires_grad=True)
    pt_out = pt_a.sigmoid()
    pt_out.backward(torch.ones_like(pt_out))

    assert_allclose(t_out.data, pt_out.detach().numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(t_a.grad, pt_a.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore
