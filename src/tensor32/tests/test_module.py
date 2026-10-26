import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from torch import nn, optim

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
from tensor32.optimizer import SGD, Adam, Optimizer, RMSprop, SGDwM


def create_linear_relu() -> tuple[Sequential, nn.Sequential]:
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

    return model, pt_model


def create_linear_tanh() -> tuple[Sequential, nn.Sequential]:
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 8),
        Tanh(),
        Linear(8, 1),
    )
    pt_model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
    )
    pt_model.load_state_dict(model.state_dict())

    return model, pt_model


def create_linear_sigmoid() -> tuple[Sequential, nn.Sequential]:
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

    return model, pt_model


RTOL = 1e-4
ATOL = 1e-4
INFERENCE_SUITE = [
    create_linear_relu(),
    create_linear_tanh(),
    create_linear_sigmoid(),
]
LINEAR_TRAINING_SUITE = [
    (
        (models_1 := create_linear_relu()),
        MSELoss(),
        nn.MSELoss(),
        SGD(models_1[0].parameters()),
        optim.SGD(models_1[1].parameters()),
    ),
    (
        (models_2 := create_linear_tanh()),
        MSELoss(),
        nn.MSELoss(),
        SGD(models_2[0].parameters(), lr=0.01),
        optim.SGD(models_2[1].parameters(), lr=0.01),
    ),
    (
        (models_3 := create_linear_sigmoid()),
        BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss(),
        SGDwM(models_3[0].parameters()),
        optim.SGD(models_3[1].parameters(), momentum=0.9),
    ),
    (
        (models_4 := create_linear_relu()),
        MSELoss(),
        nn.MSELoss(),
        RMSprop(models_4[0].parameters()),
        optim.RMSprop(models_4[1].parameters()),
    ),
    (
        (models_5 := create_linear_tanh()),
        MSELoss(),
        nn.MSELoss(),
        RMSprop(models_5[0].parameters(), lr=0.01),
        optim.RMSprop(models_5[1].parameters(), lr=0.01),
    ),
    (
        (models_6 := create_linear_sigmoid()),
        BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss(),
        RMSprop(models_6[0].parameters(), alpha=0.9),
        optim.RMSprop(models_6[1].parameters(), alpha=0.9),
    ),
    (
        (models_7 := create_linear_relu()),
        MSELoss(),
        nn.MSELoss(),
        RMSprop(models_7[0].parameters(), eps=1e-7),
        optim.RMSprop(models_7[1].parameters(), eps=1e-7),
    ),
    (
        (models_8 := create_linear_relu()),
        MSELoss(),
        nn.MSELoss(),
        Adam(models_8[0].parameters()),
        optim.Adam(models_8[1].parameters()),
    ),
    (
        (models_9 := create_linear_tanh()),
        MSELoss(),
        nn.MSELoss(),
        Adam(models_9[0].parameters(), lr=0.01),
        optim.Adam(models_9[1].parameters(), lr=0.01),
    ),
    (
        (models_10 := create_linear_sigmoid()),
        BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss(),
        Adam(models_10[0].parameters(), betas=(0.99, 0.99)),
        optim.Adam(models_10[1].parameters(), betas=(0.99, 0.99)),
    ),
    (
        (models_11 := create_linear_relu()),
        MSELoss(),
        nn.MSELoss(),
        Adam(models_11[0].parameters(), eps=1e-7),
        optim.Adam(models_11[1].parameters(), eps=1e-7),
    ),
]


@pytest.mark.parametrize("model, pt_model", INFERENCE_SUITE)
def test_inference(model: Sequential, pt_model: nn.Sequential) -> None:
    rng = np.random.default_rng()
    arr = rng.standard_normal((1, 2), dtype=np.float32)
    X = Container(data=arr)
    pt_X = torch.from_numpy(arr)

    yc = model(X, grad=False)
    assert not isinstance(yc, Tensor)
    with torch.no_grad():
        pt_yc = pt_model(pt_X)
        assert_allclose(yc.data, pt_yc.numpy(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "models, loss_fn, pt_loss_fn, optimizer, pt_optimizer", LINEAR_TRAINING_SUITE
)
def test_linear_train(
    models: tuple[Sequential, nn.Sequential],
    loss_fn,
    pt_loss_fn,
    optimizer: Optimizer,
    pt_optimizer: optim.Optimizer,
) -> None:
    model = models[0]
    pt_model = models[1]

    rng = np.random.default_rng()
    features = [
        rng.standard_normal((1, 2), dtype=np.float32),
        rng.standard_normal((1, 2), dtype=np.float32),
        rng.standard_normal((1, 2), dtype=np.float32),
    ]
    labels = [
        rng.standard_normal((1, 1), dtype=np.float32),
        rng.standard_normal((1, 1), dtype=np.float32),
        rng.standard_normal((1, 1), dtype=np.float32),
    ]

    for feature, label in zip(features, labels):
        X = Container(data=feature)
        pt_X = torch.from_numpy(feature)
        y = Container(data=label)
        pt_y = torch.from_numpy(label)

        yc = model(X)
        pt_yc = pt_model(pt_X)
        assert_allclose(yc.data, pt_yc.detach().numpy(), rtol=RTOL, atol=ATOL)

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
