"""认人：YuNet 检测 + SFace 特征比对。

    python face.py          实时认人（窗口里画框，控制台打印）
    python face.py 张三      把镜头前的人登记成"张三"（采集 5 张求平均后存底库）

底库和模型都在项目根：faces.npz、models/。
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np

from frames import open_frames

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DET_MODEL = MODELS / "yunet.onnx"
REC_MODEL = MODELS / "sface.onnx"
GALLERY = ROOT / "faces.npz"

URL = "rtsp://admin:laozhongyi1@192.168.1.14:554/stream1"

MAX_WIDTH = 640          # 检测前把整帧缩到这个宽度，1440p 也跟得上 12fps
DET_THRESHOLD = 0.6      # YuNet 置信度阈值
MATCH_THRESHOLD = 0.363  # SFace 余弦相似度阈值（官方示例值）
DISPLAY_WIDTH = 960      # 窗口里显示的宽度，原图 2560 太宽
SAMPLES = 5              # 登记时采集几张求平均
FEAT_DIM = 128           # SFace 输出维度


class Detector:
    """整帧缩小后检测，再把坐标换算回原图，省掉在 1440p 上直接跑模型的浪费。"""

    def __init__(self, max_width: int = MAX_WIDTH, threshold: float = DET_THRESHOLD):
        self._net = cv2.FaceDetectorYN.create(
            str(DET_MODEL), "", (320, 320), threshold, 0.3, 5000
        )
        self.max_width = max_width

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        scale = min(1.0, self.max_width / w)
        if scale < 1.0:
            small = cv2.resize(
                image,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        self._net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._net.detect(small)
        if faces is None:
            return []

        # 每行 15 个数：框 4 个 + 5 个关键点的 10 个 + 分数 1 个。
        # 只缩放前 14 个，分数（下标 14）别动。
        faces[:, :14] /= scale
        return list(faces)


class Embedder:
    """对齐人脸后取 128 维特征；单位化后两个特征点积就是余弦相似度。"""

    def __init__(self):
        self._net = cv2.FaceRecognizerSF.create(str(REC_MODEL), "")

    def embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self._net.alignCrop(image, face.reshape(1, -1))
        feat = self._net.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm else feat


class Gallery:
    """名字 ↔ 特征向量，存在项目根（和 models/ 放一起，不塞进代码目录）。"""

    def __init__(self, path: Path = GALLERY):
        self.path = path
        self.names: list[str] = []
        self.feats = np.zeros((0, FEAT_DIM), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = np.load(self.path)
        self.names = [str(n) for n in data["names"]]
        self.feats = data["feats"].astype(np.float32)

    def save(self) -> None:
        names = np.array(self.names, dtype=str) if self.names else np.array([], dtype="<U1")
        np.savez(self.path, names=names, feats=self.feats)

    def add(self, name: str, feat: np.ndarray) -> None:
        """同名再登记就是覆盖，方便重新录一遍。"""
        feat = feat.reshape(1, -1)
        if name in self.names:
            self.feats[self.names.index(name)] = feat
            return
        self.names.append(name)
        self.feats = np.vstack([self.feats, feat]) if len(self.feats) else feat

    def best(self, feat: np.ndarray) -> tuple[str | None, float]:
        """返回 (名字, 相似度)；低于阈值就是陌生人，名字给 None。"""
        if not self.names:
            return None, 0.0
        scores = self.feats @ feat.reshape(-1)      # 都单位化了，点积即余弦
        i = int(np.argmax(scores))
        score = float(scores[i])
        return (self.names[i] if score >= MATCH_THRESHOLD else None), score


def enroll(name: str, url: str = URL, samples: int = SAMPLES, timeout: float = 60.0) -> None:
    """从摄像头采几张最大人脸，求平均特征后登记。"""
    detector, embedder = Detector(), Embedder()
    print(f"请正对镜头，采集 {samples} 张（最多等 {timeout:.0f} 秒）…", flush=True)

    feats: list[np.ndarray] = []
    deadline = time.monotonic() + timeout
    for frame in open_frames(url):
        faces = detector.detect(frame.image)
        if faces:
            face = max(faces, key=lambda f: float(f[2] * f[3]))   # 挑面积最大的那张脸
            feats.append(embedder.embed(frame.image, face))
            print(
                f"  采到 {len(feats)}/{samples}：分数 {face[-1]:.2f}，"
                f"人脸 {int(face[2])}x{int(face[3])}",
                flush=True,
            )
            if len(feats) >= samples:
                break
        if time.monotonic() > deadline:
            raise SystemExit(f"超时，只采到 {len(feats)}/{samples} 张，确认有人在镜头前再试")

    mean = np.mean(feats, axis=0).astype(np.float32)
    mean /= np.linalg.norm(mean)

    gallery = Gallery()
    gallery.add(name, mean)
    gallery.save()
    print(f"已登记 {name}；底库共 {len(gallery.names)} 人：{', '.join(gallery.names)}")


def live(url: str = URL) -> None:
    """循环认人：框出每张脸，控制台只在结果变化时打印。"""
    detector, embedder, gallery = Detector(), Embedder(), Gallery()
    if not gallery.names:
        print("底库是空的：先运行 `python face.py 名字` 登记一张脸", flush=True)

    show = True
    try:
        cv2.namedWindow("face")
    except cv2.error:
        show = False
        print("没有可用的图形界面，只在控制台打印识别结果", flush=True)

    last: tuple[str | None, int] | None = None
    try:
        for frame in open_frames(url):
            image = frame.image
            faces = detector.detect(image)

            if show:
                h, w = image.shape[:2]
                k = DISPLAY_WIDTH / w
                view = cv2.resize(
                    image, (DISPLAY_WIDTH, int(h * k)), interpolation=cv2.INTER_AREA
                )

            for face in faces:
                name, score = gallery.best(embedder.embed(image, face))
                hit = (name, int(score * 100))          # 只在结果变化时打印，避免刷屏
                if hit != last:
                    last = hit
                    who = f"认出 {name}" if name else "不认识"
                    print(f"{who} 相似度 {score:.2f} | 帧 {frame.seq} 延迟 {frame.age:.3f}s", flush=True)

                if show:
                    x, y, fw, fh = (face[:4] * k).astype(int)
                    color = (0, 255, 0) if name else (0, 0, 255)
                    cv2.rectangle(view, (x, y), (x + fw, y + fh), color, 2)
                    label = f"{name or 'unknown'} {score:.2f}"
                    # cv2.putText 只认 ASCII，中文名字画不出来，窗口里只显示分数
                    if not label.isascii():
                        label = f"{score:.2f}"
                    cv2.putText(
                        view, label, (x + 4, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
                    )

            if show:
                cv2.imshow("face", view)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        enroll(sys.argv[1])
    else:
        live()
