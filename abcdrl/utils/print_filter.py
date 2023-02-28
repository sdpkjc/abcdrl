from __future__ import annotations

from typing import Any, Callable, Generator

import wrapt


class Filter:
    @classmethod
    def decorator(cls) -> Callable[..., Generator[dict[str, Any], None, None]]:
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs) -> Generator[dict[str, Any], None, None]:
            gen = wrapped(*args, **kwargs)
            for log_data in gen:
                if "logs" in log_data and log_data["log_type"] != "train":
                    yield log_data

        return wrapper
