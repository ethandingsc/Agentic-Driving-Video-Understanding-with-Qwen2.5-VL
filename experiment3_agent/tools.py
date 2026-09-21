from __future__ import annotations

from pathlib import Path
from typing import Any

# 导入我们已经写好并测试成功的视频处理函数
from video_tools import (
    video_metadata,
    coarse_video_frames,
    segment_frames,
    save_frames,
)

# 导入我们已经写好并测试成功的 RAG 检索器
from rag import SafetyRetriever


# ============================================================
# 1. 初始化 RAG
# ============================================================

# knowledge_base.json 和 tools.py 位于同一个文件夹
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

# 创建一个全局的驾驶安全知识检索器
# 后面 Agent 每次需要查询安全知识时，直接使用它即可
retriever = SafetyRetriever(KB_PATH)


# ============================================================
# 2. 视频信息工具
# ============================================================

def get_video_info(video_path: str) -> dict[str, Any]:
    """
    获取视频的基本信息。

    Agent 在正式分析视频之前，可以先调用这个工具，
    知道视频有多长、FPS是多少、总共有多少帧。

    返回示例：
    {
        "total_frames": 135,
        "fps": 30.0,
        "duration_sec": 4.5,
        "width": 852,
        "height": 480
    }
    """

    return video_metadata(video_path)


# ============================================================
# 3. 整段视频粗采样工具
# ============================================================

def sample_coarse_video(
    video_path: str,
    k: int = 8,
    output_dir: str = "agent_output/coarse",
) -> dict[str, Any]:
    """
    对整个视频进行粗粒度采样。

    使用场景：
    Agent 第一次拿到视频时，并不知道哪里最重要，
    所以先从整个视频中均匀抽取少量帧进行初步观察。

    例如：
    一个 30 秒的视频先均匀抽取 8 帧。

    后续 Qwen2.5-VL 可以根据这些帧判断：
    哪一个时间段可能存在危险事件。
    """

    # 获取视频信息
    meta = video_metadata(video_path)

    # 对整个视频进行均匀采样
    frames = coarse_video_frames(
        video_path,
        k=k,
    )

    # 将抽取出来的帧保存到本地
    frame_paths = save_frames(
        frames,
        output_dir,
    )

    # 把帧编号转换成对应的视频时间
    sampled_frames = []

    for (frame_index, _), frame_path in zip(frames, frame_paths):
        sampled_frames.append(
            {
                "frame_index": frame_index,
                "time_sec": round(frame_index / meta["fps"], 3),
                "path": frame_path,
            }
        )

    return {
        "sampling_type": "coarse",
        "video_path": video_path,
        "duration_sec": meta["duration_sec"],
        "sampled_frames": sampled_frames,
    }


# ============================================================
# 4. 指定时间段重新采样工具
# ============================================================

def sample_video_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    k: int = 8,
    output_dir: str = "agent_output/segment",
) -> dict[str, Any]:
    """
    对视频中的指定时间段进行重新采样。

    这是实验三中非常重要的工具。

    例如：
    第一次粗采样以后，Qwen 认为：

        1.5 秒 ~ 3.0 秒之间可能有行人横穿。

    Agent 就可以调用这个工具：

        sample_video_segment(
            video_path,
            start_sec=1.5,
            end_sec=3.0
        )

    对这个局部区域重新抽取更多帧，
    让 Qwen2.5-VL 进行更细致的第二次分析。
    """

    # 获取视频信息，主要需要 FPS
    meta = video_metadata(video_path)

    # 简单检查时间范围是否合法
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")

    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    # 防止 Agent 给出的结束时间超过视频长度
    end_sec = min(
        end_sec,
        meta["duration_sec"],
    )

    # 对指定时间段进行重新采样
    frames = segment_frames(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=meta["fps"],
        k=k,
    )

    # 保存抽取出来的帧
    frame_paths = save_frames(
        frames,
        output_dir,
    )

    sampled_frames = []

    for (frame_index, _), frame_path in zip(frames, frame_paths):
        sampled_frames.append(
            {
                "frame_index": frame_index,
                "time_sec": round(frame_index / meta["fps"], 3),
                "path": frame_path,
            }
        )

    return {
        "sampling_type": "segment",
        "video_path": video_path,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sampled_frames": sampled_frames,
    }


# ============================================================
# 5. RAG 驾驶安全知识检索工具
# ============================================================

def retrieve_safety_knowledge(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    从驾驶安全知识库中检索与当前事件最相关的知识。

    例如 Qwen 分析视频后发现：

        "A pedestrian is crossing in front of the vehicle."

    Agent 可以调用：

        retrieve_safety_knowledge(
            "pedestrian crossing safety"
        )

    RAG 会从 knowledge_base.json 中找到最相关的内容，
    例如：

        Vulnerable road user interaction
        Crossing object
        Intersection safety

    后续这些知识会和视频分析结果一起提供给 Qwen，
    用于生成最终的安全分析报告。
    """

    return retriever.search(
        query=query,
        top_k=top_k,
    )