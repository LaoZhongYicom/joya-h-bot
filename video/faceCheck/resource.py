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
        """TODO: 起 daemon 线程跑 _run，立刻返回。重复调用要幂等。"""
        raise NotImplementedError

    def stop(self, timeout: float = 5.0) -> None:
        """TODO: set 停止标志 -> notify_all 叫醒 read() -> join。

        坑：不要在这里 release() self._capture。跨线程释放正在 read()
        raise NotImplementedError

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- 消费 ----

    def read(self, timeout: float | None = None) -> Frame | None:
        """ TODO: 阻塞到出现 seq > 进场时 seq 的帧，返回它；超时或停止返回 None。

        with self.condition:
            start_seq = self._seq           # 先记下来
            while True:
                if self._seq > start_seq:   # 严格大于，同一帧不发两次
                    self._consumed_seq = self._seq
                    return self._frame
                if self._stopping.is_set():
                    return None
                # 按 timeout 调 wait()，超时 return None

        必须记 start_seq，否则消费者会拿到旧帧反复处理。
        wait() 会释放锁，生产者才进得来，别担心死锁。
        """
        raise NotImplementedError

    # ---- 采集线程 ----

    def _run(self) -> None:
        """TODO: 开连接 -> while 循环 read + publish -> finally 释放。

            self._capture = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG, params)
            try:
                while not self._stopping.is_set():
                    ok, image = self._capture.read()
                    if not ok:
                        break
                    self._publish(image)
            finally:
                self._capture.release()
                self._capture = None

        params 里放两个超时，否则死连接会挂住很久：
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
             cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
        元素必须是 int，传 float 报 "Can't parse 'params'"。
        """
        raise NotImplementedError

    def _publish(self, image) -> None:
        """TODO: 锁内 6 行，全类的核心。

            now = time.monotonic()
            with self.condition:
                if self._consumed_seq < self._seq:
                    self._dropped += 1        # 上一帧没人取
                self._seq += 1
                self._frame = Frame(image, self._seq, now)
                self.condition.notify_all()   # 必须在锁内

        「丢帧」就是 self._frame = ... 这行：旧 Frame 引用归零立刻被回收，
        没有 drop() 调用。notify_all 放锁外会丢唤醒。
        """
        raise NotImplementedError