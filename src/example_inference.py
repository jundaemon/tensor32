from tensor32 import Container, load
from tensor32.module import Linear, ReLU, Sequential

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
    model.load_state_dict(load("assets/model.pth"))

    for feature, label in zip(X, y):
        pred = model(feature, grad=False).sigmoid()
        print(f"actual: {label.data}, predicted: {pred.data}")
