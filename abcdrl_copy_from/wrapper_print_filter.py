from __future__ import annotations

from typing import Any, Callable, Generator

from combine_signatures.combine_signatures import combine_signatures


def wrapper_print_filter(
    wrapped: Callable[..., Generator[dict[str, Any], None, None]]
) -> Callable[..., Generator[dict[str, Any], None, None]]:
    @combine_signatures(wrapped)
    def _wrapper(*args, **kwargs) -> Generator[dict[str, Any], None, None]:
        gen = wrapped(*args, **kwargs)
        for log_data in gen:
            if "logs" in log_data and log_data["log_type"] != "train":
                yield log_data

    return _wrapper
