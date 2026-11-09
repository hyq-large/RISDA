import torch

from contextlib import contextmanager

ENABLE_CHECKPOINTING = True


@contextmanager
def disable_checkpointing():
    global ENABLE_CHECKPOINTING
    try:
        old, ENABLE_CHECKPOINTING = ENABLE_CHECKPOINTING, False
        yield
    finally:
        ENABLE_CHECKPOINTING = old


ENABLE_XFA = True


@contextmanager
def disable_xfa():
    global ENABLE_XFA
    try:
        old, ENABLE_XFA = ENABLE_XFA, False
        yield
    finally:
        ENABLE_XFA = old


class skip_init(torch.overrides.TorchFunctionMode):
    r"""Creates a context in which weight initialization is skipped.

    Example:
        >>> with skip_init():
        ...    layer = nn.Linear(3, 5)
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        else:
            return func(*args, **kwargs)
