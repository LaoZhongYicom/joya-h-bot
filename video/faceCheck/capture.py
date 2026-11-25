import threading
import time
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Frame:
    image: "cv2.typing.MatLike"
    seq: int
    captured_at: float  # time.monotonic()

    @property
    def age(self) -> float:
        return time.monotonic() - self.captured_at


class Resource:
    def __init__(self, path: str):
        self.path = path
        self.condition = threading.Condition()
        self._frame = None
        self._seq = 0
        self._consumed_seq = 0      # 消费者最后取走的序号，用来算丢弃
        self._stopping = threading.Event()
        self._thread = None
        self._capture = None
        self._dropped = 0

    # ---- 生命周期 ----

    def start(self) -> "Resource":
        if self._thread is not None and self._thread.is_alive():
            return self                      # 幂等
        self._stopping.clear()               # 必须复位，否则重启无效
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()                 # 叫醒 event.wait()
        with self.condition:
            self.condition.notify_all()      # 叫醒 condition.wait()
        # 不在这里 release capture：跨线程释放正在 read() 的对象会崩进程
        if self._thread is not None:
            self._thread.join(timeout)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    @property
    def dropped(self) -> int:
        return self._dropped

    # ---- 消费 ----

    def read(self, timeout: float | None = None) -> Frame | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            start_seq = self._seq             # 进场时的序号
            while True:
                if self._seq > start_seq:     # 严格大于，同一帧不发两次
                    self._consumed_seq = self._seq
                    return self._frame
                if self._stopping.is_set():
                    return None
                if deadline is None:
                    self.condition.wait()     # 会释放锁，生产者才进得来
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

    # ---- 采集线程 ----

    def _run(self) -> None:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ]
        capture = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG, params)
        if not capture.isOpened():
            capture.release()
            self._finish()
            return

        self._capture = capture
        try:
            while not self._stopping.is_set():
                ok, image = capture.read()
                if not ok:
                    break                     # 流断了或文件读完了
                self._publish(image)
        finally:
            self._capture = None
            capture.release()
            self._finish()

    def _finish(self) -> None:
        """采集线程结束时叫醒所有阻塞的 read()，让它们返回 None。"""
        self._stopping.set()
        with self.condition:
            self.condition.notify_all()

    def _publish(self, image) -> None:
        now = time.monotonic()
        with self.condition:
            if self._consumed_seq < self._seq:
                self._dropped += 1            # 上一帧没人取
            self._seq += 1
            self._frame = Frame(image, self._seq, now)   # 覆盖即丢弃
            self.condition.notify_all()       # 必须在锁内