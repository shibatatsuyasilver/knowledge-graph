"""Utilities for calling functions with version-compatible keyword arguments."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict


def call_with_compatible_kwargs(
    func: Callable[..., Any],
    first_arg: str,
    **overrides: Any,
) -> Any:
    """Call ``func(first_arg, **filtered_kwargs)`` keeping only supported params.

    Filters ``overrides`` to the params actually declared by ``func``, so that
    monkeypatched or older-signature versions don't raise ``TypeError``.
    ``None`` values in ``overrides`` are excluded before the signature check.

    Args:
        func: The callable to invoke.
        first_arg: The first positional argument (typically a question string).
        **overrides: Keyword arguments to pass if the function supports them.

    Returns:
        The return value of ``func``.
    """
    kwargs: Dict[str, Any] = {k: v for k, v in overrides.items() if v is not None}

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(first_arg, **kwargs)

    params = signature.parameters
    supports_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if supports_var_keyword:
        return func(first_arg, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return func(first_arg, **filtered)
