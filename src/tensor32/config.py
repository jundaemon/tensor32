from typing import Any


class Config:
    grad_enabled: bool = True


class no_grad:
    def __enter__(self) -> None:
        self.prev_state = Config.grad_enabled
        Config.grad_enabled = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        Config.grad_enabled = self.prev_state
