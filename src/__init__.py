"""ADFNet: Task-completion-driven adaptive fatigue modeling."""

__all__ = ["ADFNet"]


def __getattr__(name: str):
    if name == "ADFNet":
        from models.adfnet import ADFNet

        return ADFNet
    raise AttributeError(name)
