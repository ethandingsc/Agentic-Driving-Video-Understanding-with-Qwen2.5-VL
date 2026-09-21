from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import requests

from .config import AgentConfig


# ============================================================
# 1. 图片转换
# ============================================================
# 将本地图片转换成 Base64 Data URL。
# 以后发送给 AutoDL 上的 Qwen2.5-VL 时会用到。
def _image_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"

    data = Path(path).read_bytes()

    return (
        f"data:{mime};base64,"
        + base64.b64encode(data).decode("ascii")
    )


# ============================================================
# 2. 从模型输出中提取 JSON
# ============================================================
# Qwen 有时会返回：
#
# {
#   ...
# }
#
# 也可能返回：
#
# ```json
# {
#   ...
# }
# ```
#
# 这个函数负责把真正的 JSON 提取出来。
def _extract_json(text: str) -> Any:
    text = (text or "").strip()

    candidates = [text]

    # 尝试提取 Markdown code block 中的内容
    fenced = re.findall(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    candidates.extend(fenced)

    # 直接尝试解析
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 如果前后还有其他文字，就尝试寻找 {...} 或 [...]
    for opening, closing in [
        ("{", "}"),
        ("[", "]"),
    ]:
        start = text.find(opening)
        end = text.rfind(closing)

        if start >= 0 and end > start:
            try:
                return json.loads(
                    text[start : end + 1]
                )
            except json.JSONDecodeError:
                pass

    raise ValueError(
        f"Model did not return valid JSON: {text[:500]}"
    )


# ============================================================
# 3. Qwen / SGLang Client
# ============================================================

class SGLangClient:

    def __init__(
        self,
        config: AgentConfig,
    ):
        """
        初始化模型客户端。

        当前阶段：
            AGENT_MOCK=1
            → 不连接 AutoDL
            → 使用假的 VLM 输出

        以后：
            AGENT_MOCK=0
            → 通过 HTTP API
            → 调用 AutoDL 上的 SGLang
            → SGLang 再调用 Qwen2.5-VL
        """

        self.config = config

        # 是否使用 Mock 模式
        self.mock = os.getenv(
            "AGENT_MOCK",
            "0",
        ) == "1"

    # ========================================================
    # 4. 普通 Chat
    # ========================================================

    def chat(
        self,
        text: str,
        image_paths: list[str] | None = None,
        max_tokens: int = 768,
        system_prompt: str | None = None,
    ) -> str:
        """
        调用 VLM。

        参数：
            text
                当前任务的 User Prompt

            image_paths
                需要发送给视觉模型的图片

            system_prompt
                当前 Agent 使用的 System Prompt

        当前如果是 Mock 模式：
            不调用任何模型，直接返回模拟结果。

        非 Mock 模式：
            通过 SGLang API 调用 Qwen2.5-VL。
        """

        # ====================================================
        # Mock Mode
        # ====================================================
        if self.mock:

            # -----------------------------------------------
            # Planner 模拟结果
            # -----------------------------------------------
            if "planning step" in text.lower():
                return json.dumps(
                    {
                        "segments": [
                            {
                                "start_sec": 1.0,
                                "end_sec": 3.0,
                                "priority": "high",
                                "reason": "A pedestrian-road interaction may require closer inspection.",
                                "topics": [
                                    "pedestrian",
                                    "crossing",
                                ],
                            }
                        ]
                    }
                )

            # -----------------------------------------------
            # Segment Analysis 模拟结果
            # -----------------------------------------------
            if "selected temporal segment" in text.lower():
                return json.dumps(
                    {
                        "observation": (
                            "A pedestrian is visible near "
                            "the roadway and crosses the "
                            "vehicle's path."
                        ),
                        "road_users": [
                            "vehicle",
                            "pedestrian",
                        ],
                        "risk_factors": [
                            "pedestrian crossing",
                            "potential path conflict",
                        ],
                        "event_sequence": [
                            "Vehicle approaches the crossing area.",
                            "Pedestrian enters the roadway.",
                            "Vehicle continues approaching the crossing.",
                        ],
                        "uncertainty": [
                            "Exact vehicle speed cannot be determined from the frames.",
                            "Exact distance between the vehicle and pedestrian cannot be determined.",
                        ],
                        "safety_topics": [
                            "pedestrian",
                            "crossing_object",
                            "intersection",
                        ],
                        "assessment": (
                            "The frames show a pedestrian "
                            "crossing interaction that warrants "
                            "closer safety review."
                        ),
                    }
                )

            # -----------------------------------------------
            # Final Report 模拟结果
            # -----------------------------------------------
            if "final synthesis step" in text.lower():
                return (
                    "1. Executive Summary\n"
                    "A pedestrian crossing interaction "
                    "was identified for offline safety review.\n\n"

                    "2. Key Events and Evidence\n"
                    "The pedestrian enters the roadway "
                    "while the ego vehicle approaches.\n\n"

                    "3. Safety Risk Analysis\n"
                    "The interaction creates a potential "
                    "vehicle-pedestrian conflict.\n\n"

                    "4. Relevant Safety Guidance\n"
                    "Retrieved guidance emphasizes caution "
                    "around pedestrians and crossing areas.\n\n"

                    "5. Uncertainty / Limitations\n"
                    "Exact speed, distance, and driver intent "
                    "cannot be determined from the sampled frames.\n\n"

                    "6. Follow-up Review Suggestions\n"
                    "Run the case with the real Qwen2.5-VL "
                    "model and inspect the selected frames."
                )

            # -----------------------------------------------
            # 兜底 Mock 输出
            # -----------------------------------------------
            return json.dumps(
                {
                    "observation": "Mock visual analysis",
                    "road_users": [
                        "vehicle",
                        "pedestrian",
                    ],
                    "risk_factors": [
                        "potential conflict",
                    ],
                    "event_sequence": [
                        "approach",
                        "interaction",
                    ],
                    "uncertainty": [
                        "mock mode",
                    ],
                    "safety_topics": [
                        "pedestrian",
                        "crossing",
                    ],
                    "assessment": (
                        "Candidate interaction should be "
                        "reviewed with the real VLM."
                    ),
                }
            )

        # ====================================================
        # Real SGLang Mode
        # ====================================================

        # 构建多模态 content
        content: list[dict[str, Any]] = []

        # 添加图片
        for path in image_paths or []:

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(path)
                    },
                }
            )

        # 添加文字 Prompt
        content.append(
            {
                "type": "text",
                "text": text,
            }
        )

        # 如果没有单独传 System Prompt，
        # 就使用一个最简单的默认提示。
        if system_prompt is None:
            system_prompt = (
                "You are an offline driving-video "
                "safety analysis agent."
            )

        # ====================================================
        # 构造 OpenAI-compatible API 请求
        # ====================================================

        payload = {
            "model": self.config.model_name,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],

            "temperature": 0.1,
            "max_tokens": max_tokens,
        }

        # 发送 HTTP 请求到 SGLang
        response = requests.post(
            self.config.api_url,
            json=payload,
            timeout=self.config.request_timeout_sec,
        )

        response.raise_for_status()

        data = response.json()

        # 提取模型最终回答
        return data["choices"][0]["message"]["content"]

    # ========================================================
    # 5. JSON Chat
    # ========================================================

    def json_chat(
        self,
        text: str,
        image_paths: list[str] | None = None,
        max_tokens: int = 768,
        system_prompt: str | None = None,
    ) -> Any:
        """
        与模型交互，并要求返回 JSON。

        主要用于：

        Planner
            ↓
        JSON

        Segment Analysis
            ↓
        JSON
        """

        result = self.chat(
            text=text,
            image_paths=image_paths,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        return _extract_json(result)