from tensor32 import Container, save
from tensor32.module import BCEWithLogitsLoss, Linear, ReLU, Sequential
from tensor32.optimizer import Adam

if __name__ == "__main__":
    X = [
        Container([[0, 0]]),
        Container([[1, 0]]),
        Container([[0, 1]]),
        Container([[1, 1]]),
    ]
    y = [
        Container([[0]]),
        Container([[1]]),
        Container([[1]]),
        Container([[0]]),
    ]
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())
    epochs = 50_000

    for epoch in range(epochs):
        for feature, label in zip(X, y):
            yc = model(feature)
            loss = loss_fn(yc, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    save(model.state_dict(), "assets/model.pth")
