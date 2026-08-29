"""Model registry. Each model lives in its own file and registers a name here.

Add a model:
    1. Create ``models/<yourmodel>.py`` subclassing ``Model`` and decorated
       ``@register("name")``.
    2. Import it at the bottom of this file so the decorator runs.
    3. Run it via ``python -m tippytop run --model <name>``.
"""
from .base import Model

# name -> factory (the class). Import lazily inside a factory if it needs heavy deps.
MODELS: dict[str, callable] = {}


def register(name: str):
    def deco(factory):
        MODELS[name] = factory
        return factory
    return deco


def build(name: str, **kwargs) -> Model:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


# Import model modules so their @register decorators populate MODELS.
# (Placed after register/build are defined to avoid a circular import.)
from . import fm, popularity, random_model, fm_rank, ensemble  # noqa: E402,F401

__all__ = ["Model", "MODELS", "register", "build"]
