from __future__ import annotations

from typing import Iterator

import time

from resource import Frame, Resource


def frames(res: Resource, timeout: float | None = 1.0) -> Iterator[Frame]:
    while True:
        started = time.monotonic()
        frame = res.read(timeout=timeout)

        if frame is not None:
            yield frame
            continue
        if not res.is_alive:
            return
        if time.monotonic() - started < 0.05:
            time.sleep(0.01)