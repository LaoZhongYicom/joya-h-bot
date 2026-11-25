from __future__ import annotations

from typing import Iterator

import time

from capture import Frame, Resource


def frames(res: Resource, timeout: float | None = 2.0) -> Iterator[Frame]:
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


def open_frames(url: str, timeout: float | None = 2.0) -> Iterator[Frame]:
    """给一个地址就不断出帧；不想自己管 Resource 的生命周期就用这个。"""
    with Resource(url) as res:
        yield from frames(res, timeout)


if __name__ == "__main__":
    url = "rtsp://admin:laozhongyi1@192.168.1.14:554/stream1"

    try:
        for face in open_frames(url):
            print(face.seq, face.image.shape, round(face.age, 3), flush=True)
    except KeyboardInterrupt:
        pass
