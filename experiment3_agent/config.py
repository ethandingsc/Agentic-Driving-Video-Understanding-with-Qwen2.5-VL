from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Agent 的配置。

    这些参数主要控制：
    1. 如何连接 AutoDL 上的 SGLang
    2. 视频采样数量
    3. Agent 最多分析几个时间段
    4. RAG 返回多少条知识
    """

    # ========================================================
    # SGLang API 配置
    # ========================================================

    # 本地 Agent 通过 SSH 隧道访问 AutoDL
    # 本地 30000 → SSH Tunnel → AutoDL 30000 → SGLang
    api_url: str = os.getenv(
        "SGLANG_API_URL",
        "http://127.0.0.1:30000/v1/chat/completions",
    )

    # AutoDL 上实际加载的 Qwen2.5-VL 模型路径
    model_name: str = os.getenv(
        "MODEL_NAME",
        "/root/autodl-tmp/models/experiment2/qwen2.5vl-automingo-merged",
    )

    # ========================================================
    # 视频采样配置
    # ========================================================

    # 候选帧数量
    # 当前 Agent 没有单独使用这个参数，可以暂时保留。
    candidate_frames: int = int(
        os.getenv(
            "AGENT_CANDIDATE_FRAMES",
            "32",
        )
    )

    # 第一次浏览整个视频时抽取的帧数
    coarse_frames: int = int(
        os.getenv(
            "AGENT_COARSE_FRAMES",
            "6",
        )
    )

    # Agent 找到重点时间段以后，
    # 在局部区域重新抽取多少帧
    segment_frames: int = int(
        os.getenv(
            "AGENT_SEGMENT_FRAMES",
            "8",
        )
    )

    # Planner 最多选择多少个重点时间段
    max_segments: int = int(
        os.getenv(
            "AGENT_MAX_SEGMENTS",
            "3",
        )
    )

    # ========================================================
    # RAG 配置
    # ========================================================

    # 每次检索最多返回多少条安全知识
    top_k_docs: int = int(
        os.getenv(
            "AGENT_TOP_K_DOCS",
            "4",
        )
    )

    # ========================================================
    # API 请求配置
    # ========================================================

    # 单次请求最长等待时间
    # Qwen2.5-VL 处理多张图片可能需要一些时间
    request_timeout_sec: int = int(
        os.getenv(
            "AGENT_TIMEOUT_SEC",
            "300",
        )
    )


# ============================================================
# 默认 System Prompt
# ============================================================
# 注意：
# 我们现在已经在 prompts.py 中定义了更加完整的
# SYSTEM_PROMPT。
#
# 所以这个 DEFAULT_SYSTEM_PROMPT 主要作为备用配置，
# 防止某些地方没有传入 SYSTEM_PROMPT 时程序报错。

DEFAULT_SYSTEM_PROMPT = """
You are an offline driving-video safety analysis agent.

Analyze recorded dashcam footage for:
- accident and near-miss events
- vulnerable road users
- vehicle cut-ins and lane changes
- leading-vehicle braking
- merging interactions
- traffic signals and intersections
- crossing objects
- construction zones
- other potentially hazardous or ambiguous situations

Base your analysis on visible evidence.
Do not invent events, timestamps, intent, speed, or distance
that cannot be supported by the provided frames.

This system is for offline engineering analysis only.
It is not used for real-time vehicle control or autonomous-driving decisions.
""".strip()