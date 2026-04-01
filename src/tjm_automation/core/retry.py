from __future__ import annotations

from collections.abc import Callable
from time import sleep
from typing import TypeVar


T = TypeVar("T")


def retry(
    func: Callable[[], T],
    *,
    attempts: int,
    delay_seconds: float,
    retry_on: type[Exception] = Exception,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retry_on as exc:
            last_error = exc
            if attempt == attempts:
                break
            sleep(delay_seconds)
    assert last_error is not None
    raise last_error

