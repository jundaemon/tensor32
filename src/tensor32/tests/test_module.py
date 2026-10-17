import torch
from numpy.testing import assert_allclose, assert_array_equal
from torch import nn, optim, tensor

from tensor32 import Container, Tensor
from tensor32.module import (
    BCEWithLogitsLoss,
    Linear,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from tensor32.optimizer import SGD, Adam, RMSprop

RTOL = 1e-4
ATOL = 1e-4


def test_inference_1() -> None:
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
        ReLU(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.ReLU(),
    )
    pt_model.load_state_dict(model.state_dict())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X, grad=False)
    assert not isinstance(yc, Tensor)
    with torch.no_grad():
        pt_yc = pt_model(pt_X)
        assert_allclose(yc.data, pt_yc.numpy(), rtol=RTOL, atol=ATOL)


def test_inference_2() -> None:
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 8),
        Tanh(),
        Linear(8, 1),
        Tanh(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Tanh(),
    )
    pt_model.load_state_dict(model.state_dict())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X, grad=False)
    assert not isinstance(yc, Tensor)
    with torch.no_grad():
        pt_yc = pt_model(pt_X)
        assert_allclose(yc.data, pt_yc.numpy(), rtol=RTOL, atol=ATOL)


def test_inference_3() -> None:
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 8),
        Sigmoid(),
        Linear(8, 1),
        Sigmoid(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    pt_model.load_state_dict(model.state_dict())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X, grad=False)
    assert not isinstance(yc, Tensor)
    with torch.no_grad():
        pt_yc = pt_model(pt_X)
        assert_allclose(yc.data, pt_yc.numpy(), rtol=RTOL, atol=ATOL)


def test_train_1() -> None:
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
        ReLU(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.ReLU(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = SGD(model.parameters(), lr=0.01)
    pt_optimizer = optim.SGD(pt_model.parameters(), lr=0.01)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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


def test_train_2() -> None:
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 8),
        Tanh(),
        Linear(8, 1),
        Tanh(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Tanh(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = SGD(model.parameters(), momentum=0.9)
    pt_optimizer = optim.SGD(pt_model.parameters(), momentum=0.9)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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


def test_train_3() -> None:
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 8),
        Sigmoid(),
        Linear(8, 1),
        Sigmoid(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = SGD(model.parameters())
    pt_optimizer = optim.SGD(pt_model.parameters())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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


def test_train_4() -> None:
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = BCEWithLogitsLoss()
    pt_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = SGD(model.parameters())
    pt_optimizer = optim.SGD(pt_model.parameters())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

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


def test_train_5() -> None:
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
        ReLU(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.ReLU(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = RMSprop(model.parameters(), lr=0.01)
    pt_optimizer = optim.RMSprop(pt_model.parameters(), lr=0.01)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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


def test_train_6() -> None:
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 8),
        Tanh(),
        Linear(8, 1),
        Tanh(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Tanh(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = RMSprop(model.parameters(), alpha=0.9)
    pt_optimizer = optim.RMSprop(pt_model.parameters(), alpha=0.9)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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


def test_train_7() -> None:
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 8),
        Sigmoid(),
        Linear(8, 1),
        Sigmoid(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = RMSprop(model.parameters(), eps=1e-7)
    pt_optimizer = optim.RMSprop(pt_model.parameters(), eps=1e-7)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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


def test_train_8() -> None:
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
        ReLU(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.ReLU(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = BCEWithLogitsLoss()
    pt_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = RMSprop(model.parameters())
    pt_optimizer = optim.RMSprop(pt_model.parameters())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

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


def test_train_9() -> None:
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 8),
        ReLU(),
        Linear(8, 1),
        ReLU(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.ReLU(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=0.01)
    pt_optimizer = optim.Adam(pt_model.parameters(), lr=0.01)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_2 = Container([[3, 4]])
    pt_X_2 = tensor([[3, 4]], dtype=torch.float32)

    yc_2 = model(X_2)
    pt_yc_2 = pt_model(pt_X_2)
    assert_allclose(yc_2.data, pt_yc_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_2 = Container([[2]])
    pt_y_2 = tensor([[2]], dtype=torch.float32)
    loss_2 = loss_fn(yc_2, y_2)
    pt_loss_2 = pt_loss_fn(pt_yc_2, pt_y_2)
    assert_allclose(loss_2.data, pt_loss_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_2.backward()
    pt_loss_2.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_3 = Container([[2, 3]])
    pt_X_3 = tensor([[2, 3]], dtype=torch.float32)

    yc_3 = model(X_3)
    pt_yc_3 = pt_model(pt_X_3)
    assert_allclose(yc_3.data, pt_yc_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_3 = Container([[1.5]])
    pt_y_3 = tensor([[1.5]], dtype=torch.float32)
    loss_3 = loss_fn(yc_3, y_3)
    pt_loss_3 = pt_loss_fn(pt_yc_3, pt_y_3)
    assert_allclose(loss_3.data, pt_loss_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_3.backward()
    pt_loss_3.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)


def test_train_10() -> None:
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 8),
        Tanh(),
        Linear(8, 1),
        Tanh(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Tanh(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = Adam(model.parameters(), betas=(0.99, 0.99))
    pt_optimizer = optim.Adam(pt_model.parameters(), betas=(0.99, 0.99))

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_2 = Container([[3, 4]])
    pt_X_2 = tensor([[3, 4]], dtype=torch.float32)

    yc_2 = model(X_2)
    pt_yc_2 = pt_model(pt_X_2)
    assert_allclose(yc_2.data, pt_yc_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_2 = Container([[2]])
    pt_y_2 = tensor([[2]], dtype=torch.float32)
    loss_2 = loss_fn(yc_2, y_2)
    pt_loss_2 = pt_loss_fn(pt_yc_2, pt_y_2)
    assert_allclose(loss_2.data, pt_loss_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_2.backward()
    pt_loss_2.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_3 = Container([[2, 3]])
    pt_X_3 = tensor([[2, 3]], dtype=torch.float32)

    yc_3 = model(X_3)
    pt_yc_3 = pt_model(pt_X_3)
    assert_allclose(yc_3.data, pt_yc_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_3 = Container([[1.5]])
    pt_y_3 = tensor([[1.5]], dtype=torch.float32)
    loss_3 = loss_fn(yc_3, y_3)
    pt_loss_3 = pt_loss_fn(pt_yc_3, pt_y_3)
    assert_allclose(loss_3.data, pt_loss_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_3.backward()
    pt_loss_3.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)


def test_train_11() -> None:
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 8),
        Sigmoid(),
        Linear(8, 1),
        Sigmoid(),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = MSELoss()
    pt_loss_fn = nn.MSELoss()

    optimizer = Adam(model.parameters(), eps=1e-7)
    pt_optimizer = optim.Adam(pt_model.parameters(), eps=1e-7)

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

    yc = model(X)
    pt_yc = pt_model(pt_X)
    assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

    y = Container([[1]])
    pt_y = tensor([[1]], dtype=torch.float32)
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
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_2 = Container([[3, 4]])
    pt_X_2 = tensor([[3, 4]], dtype=torch.float32)

    yc_2 = model(X_2)
    pt_yc_2 = pt_model(pt_X_2)
    assert_allclose(yc_2.data, pt_yc_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_2 = Container([[2]])
    pt_y_2 = tensor([[2]], dtype=torch.float32)
    loss_2 = loss_fn(yc_2, y_2)
    pt_loss_2 = pt_loss_fn(pt_yc_2, pt_y_2)
    assert_allclose(loss_2.data, pt_loss_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_2.backward()
    pt_loss_2.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_3 = Container([[2, 3]])
    pt_X_3 = tensor([[2, 3]], dtype=torch.float32)

    yc_3 = model(X_3)
    pt_yc_3 = pt_model(pt_X_3)
    assert_allclose(yc_3.data, pt_yc_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_3 = Container([[1.5]])
    pt_y_3 = tensor([[1.5]], dtype=torch.float32)
    loss_3 = loss_fn(yc_3, y_3)
    pt_loss_3 = pt_loss_fn(pt_yc_3, pt_y_3)
    assert_allclose(loss_3.data, pt_loss_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_3.backward()
    pt_loss_3.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)


def test_train_12() -> None:
    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 8),
        Sigmoid(),
        Linear(8, 1),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
    )
    pt_model.load_state_dict(model.state_dict())

    loss_fn = BCEWithLogitsLoss()
    pt_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters())
    pt_optimizer = optim.Adam(pt_model.parameters())

    X = Container([[1, 2]])
    pt_X = tensor([[1, 2]], dtype=torch.float32)

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
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_2 = Container([[3, 4]])
    pt_X_2 = tensor([[3, 4]], dtype=torch.float32)

    yc_2 = model(X_2)
    pt_yc_2 = pt_model(pt_X_2)
    assert_allclose(yc_2.data, pt_yc_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_2 = Container([[1]])
    pt_y_2 = tensor([[1]], dtype=torch.float32)
    loss_2 = loss_fn(yc_2, y_2)
    pt_loss_2 = pt_loss_fn(pt_yc_2, pt_y_2)
    assert_allclose(loss_2.data, pt_loss_2.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_2.backward()
    pt_loss_2.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    pt_optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)

    X_3 = Container([[2, 3]])
    pt_X_3 = tensor([[2, 3]], dtype=torch.float32)

    yc_3 = model(X_3)
    pt_yc_3 = pt_model(pt_X_3)
    assert_allclose(yc_3.data, pt_yc_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    y_3 = Container([[0]])
    pt_y_3 = tensor([[0]], dtype=torch.float32)
    loss_3 = loss_fn(yc_3, y_3)
    pt_loss_3 = pt_loss_fn(pt_yc_3, pt_y_3)
    assert_allclose(loss_3.data, pt_loss_3.detach().numpy(), rtol=RTOL, atol=ATOL)

    loss_3.backward()
    pt_loss_3.backward()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.grad, pt_param.grad.numpy(), rtol=RTOL, atol=ATOL)  # type: ignore

    optimizer.step()
    pt_optimizer.step()
    for param, pt_param in zip(model.parameters(), pt_model.parameters()):
        assert_allclose(param.data, pt_param.detach().numpy(), rtol=RTOL, atol=ATOL)

    optimizer.zero_grad()
    for param in model.parameters():
        assert_array_equal(param.grad, 0)
