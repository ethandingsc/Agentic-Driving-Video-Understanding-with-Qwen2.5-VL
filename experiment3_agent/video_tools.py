from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def video_metadata(video_path: str) -> dict[str, Any]:
    """
    读取视频的基本信息。
    Agent 后续可以利用这些信息知道视频有多长、FPS是多少等。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if total_frames <= 0:
        raise ValueError(f"Invalid frame count: {video_path}")

    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": total_frames / fps,
        "width": width,
        "height": height,
    }


def decode_frames_at_indices(
    video_path: str,
    indices: list[int],
) -> list[tuple[int, Image.Image]]:
    """
    根据指定的帧编号，从视频中真正读取这些帧。

    例如：
    indices = [0, 30, 60]
    就读取第 0、30、60 帧。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    result: list[tuple[int, Image.Image]] = []

    # 去重、排序，并保证帧编号 >= 0
    for idx in sorted(set(max(0, int(i)) for i in indices)):

        # 跳到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ok, frame = cap.read()

        if not ok:
            continue

        # OpenCV 默认是 BGR，PIL 使用 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result.append(
            (idx, Image.fromarray(frame))
        )

    cap.release()

    if not result:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    return result


def uniform_indices(
    start_frame: int,
    end_frame: int,
    k: int,
) -> list[int]:
    """
    在指定范围内均匀选择 k 帧。

    例如：
    start_frame = 0
    end_frame = 100
    k = 5

    大约得到：
    [0, 25, 50, 75, 100]

    这里就是实验二中的 Uniform Sampling 思路。
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    if k == 1:
        return [start_frame]

    return [
        int(round(x))
        for x in np.linspace(
            start_frame,
            end_frame,
            num=min(k, end_frame - start_frame + 1),
        )
    ]


def segment_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    k: int,
) -> list[tuple[int, Image.Image]]:
    """
    对视频中的某一个时间段重新进行密集采样。

    例如 Agent 认为：
    12秒 ~ 17秒存在危险

    就可以调用：
    segment_frames(video, 12, 17, fps, 8)

    在这个局部时间段重新抽取 8 帧。
    """

    # 秒 → 帧编号
    start_frame = max(
        0,
        int(round(start_sec * fps)),
    )

    end_frame = max(
        start_frame,
        int(round(end_sec * fps)),
    )

    # 在这个局部时间段进行均匀采样
    indices = uniform_indices(
        start_frame,
        end_frame,
        k,
    )

    return decode_frames_at_indices(
        video_path,
        indices,
    )


def coarse_video_frames(
    video_path: str,
    k: int,
) -> list[tuple[int, Image.Image]]:
    """
    对整个视频进行第一次粗粒度采样。

    Agent 第一次不知道哪里重要，
    所以先均匀看整个视频。

    之后如果发现某个时间段值得关注，
    再调用 segment_frames() 进行局部重新采样。
    """

    meta = video_metadata(video_path)

    indices = uniform_indices(
        0,
        meta["total_frames"] - 1,
        min(k, meta["total_frames"]),
    )

    return decode_frames_at_indices(
        video_path,
        indices,
    )


def save_frames(
    frames: list[tuple[int, Image.Image]],
    out_dir: str,
) -> list[str]:
    """
    把抽取出来的帧保存成 JPG。

    主要用于：
    1. Debug
    2. 查看 Agent 到底抽了哪些帧
    3. 后续发送给 Qwen2.5-VL
    """

    path = Path(out_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    files: list[str] = []

    for idx, image in frames:

        p = path / f"frame_{idx:06d}.jpg"

        image.save(
            p,
            quality=92,
        )

        files.append(
            str(p.resolve())
        )

    return files