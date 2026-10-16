import torch
from numpy.testing import assert_allclose, assert_array_equal
from torch import nn, optim, tensor

from tensor32 import Container
from tensor32.module import (
    BCEWithLogitsLoss,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from tensor32.optimizer import SGD

RTOL = 1e-4
ATOL = 1e-4


def test_forward() -> None:
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 6),
        Tanh(),
        Linear(6, 1),
        ReLU(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 6),
        nn.Tanh(),
        nn.Linear(6, 1),
        nn.ReLU(),
    )
    pt_model.load_state_dict(model.state_dict())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X, grad=False)
    with torch.no_grad():
        pt_yc = pt_model(pt_X)
        assert_allclose(yc.data, pt_yc.numpy(), rtol=RTOL, atol=ATOL)


def test_mlp_1() -> None:
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 5),
        Sigmoid(),
        Linear(5, 3),
        Tanh(),
    )
    pt_model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 5),
        nn.Sigmoid(),
        nn.Linear(5, 3),
        nn.Tanh(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = SGD(model.parameters())
    pt_optimizer = optim.SGD(pt_model.parameters())

    X = Container([[1, 2, 3]])
    pt_X = tensor([[1, 2, 3]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[3, 2, 1]])
    pt_y = tensor([[3, 2, 1]], dtype=torch.float32)
    loss = loss_fn(yc, y)
    pt_loss = pt_loss_fn(pt_yc, pt_y)
    assert_allclose(loss.data, pt_loss.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss.backward()
    pt_loss.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)


def test_mlp_2() -> None:
    model = Sequential(
        Linear(4, 3),
        Tanh(),
        Linear(3, 2),
        ReLU(),
        Linear(2, 1),
        Sigmoid(),
    )
    pt_model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Tanh(),
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
        nn.Sigmoid(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = SGD(model.parameters(), momentum=1e-3)
    pt_optimizer = optim.SGD(pt_model.parameters(), momentum=1e-3)

    X = Container([[1, 2, 3, 4]])
    pt_X = tensor([[1, 2, 3, 4]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[3]])
    pt_y = tensor([[3]], dtype=torch.float32)
    loss = loss_fn(yc, y)
    pt_loss = pt_loss_fn(pt_yc, pt_y)
    assert_allclose(loss.data, pt_loss.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss.backward()
    pt_loss.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)


def test_mlp_3() -> None:
    model = Sequential(
        Linear(3, 5),
        ReLU(),
        Linear(5, 7),
        Tanh(),
        Linear(7, 1),
        Sigmoid(),
    )
    pt_model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 7),
        nn.Tanh(),
        nn.Linear(7, 1),
        nn.Sigmoid(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = BCEWithLogitsLoss()
    pt_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = SGD(model.parameters())
    pt_optimizer = optim.SGD(pt_model.parameters())

    X = Container([[1.5, 2.3, 3.2]])
    pt_X = tensor([[1.5, 2.3, 3.2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[0]])
    pt_y = tensor([[0]], dtype=torch.float32)
    loss = loss_fn(yc, y)
    pt_loss = pt_loss_fn(pt_yc, pt_y)
    assert_allclose(loss.data, pt_loss.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss.backward()
    pt_loss.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)
