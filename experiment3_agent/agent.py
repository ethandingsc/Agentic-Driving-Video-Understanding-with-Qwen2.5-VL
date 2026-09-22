from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

# ============================================================
# LangGraph
# ============================================================
# LangGraph 用来把 Agent 的多个步骤串起来。
# 如果环境里没有安装 LangGraph，就使用下面的简单 fallback。
try:
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# ============================================================
# 一个非常简单的备用流程
# ============================================================
# 如果没有安装 LangGraph，就按照 nodes 列表顺序执行。
# 这样我们学习 Agent 时，不会因为缺少 LangGraph 而完全无法运行。
class _FallbackGraph:
    """没有 LangGraph 时使用的简单顺序执行器。"""

    def __init__(self, nodes):
        self.nodes = nodes

    def invoke(self, state):
        current = dict(state)

        for node in self.nodes:
            current.update(node(current))

        return current


# ============================================================
# 导入项目中的模块
# ============================================================

from .config import AgentConfig
from .model_client import SGLangClient
from .prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    SEGMENT_PROMPT,
    FINAL_REPORT_PROMPT,
)
from .rag import SafetyRetriever
from .video_tools import (
    coarse_video_frames,
    save_frames,
    segment_frames,
    video_metadata,
)


# ============================================================
# Agent State
# ============================================================
# State 就是 Agent 在不同步骤之间传递的数据。
#
# 可以理解成一个“共享工作区”：
#
# 视频
#   ↓
# metadata
#   ↓
# coarse_frames
#   ↓
# plan
#   ↓
# segment_results
#   ↓
# retrieved_docs
#   ↓
# report
#
class AgentState(TypedDict, total=False):

    # 输入视频路径
    video_path: str

    # Agent 本次运行产生的临时工作目录
    work_dir: str

    # 视频基本信息
    metadata: dict[str, Any]

    # 第一次粗采样得到的帧
    coarse_frames: list[dict[str, Any]]

    # Planner 决定需要进一步分析的时间段
    plan: list[dict[str, Any]]

    # 对重点时间段进行重新分析后的结果
    segment_results: list[dict[str, Any]]

    # RAG 找到的安全知识
    retrieved_docs: list[dict[str, Any]]

    # 最终报告
    report: str


# ============================================================
# Driving Video Agent
# ============================================================

class DrivingVideoAgent:

    def __init__(
        self,
        config: AgentConfig | None = None,
        kb_path: str | None = None,
    ):
        """
        初始化驾驶视频 Agent。
        这里主要准备三个东西：
        1. Config
           保存 Agent 的参数
        2. SGLangClient
           以后负责调用 AutoDL 上的 Qwen2.5-VL
        3. SafetyRetriever
           负责从 knowledge_base.json 检索安全知识
        """

        # 使用默认配置，或者使用用户传进来的配置
        self.config = config or AgentConfig()

        # 创建 Qwen / SGLang 客户端
        #
        self.client = SGLangClient(self.config)

        # 默认知识库就在当前目录
        default_kb = Path(__file__).with_name(
            "knowledge_base.json"
        )

        # 初始化 RAG Retriever
        self.retriever = SafetyRetriever(
            kb_path or default_kb
        )

        # 构建 Agent 工作流程
        self.graph = self._build_graph()

    # ========================================================
    # 构建 Agent Workflow
    # ========================================================

    def _build_graph(self):

        nodes = [
            self.inspect_video,
            self.plan_segments,
            self.analyze_segments,
            self.retrieve_guidance,
            self.write_report,
        ]

        # 如果没有安装 LangGraph
        # 就使用最简单的顺序执行方式
        if not HAS_LANGGRAPH:
            return _FallbackGraph(nodes)

        # ----------------------------------------------------
        # 使用 LangGraph 构建正式的 Agent workflow
        # ----------------------------------------------------

        builder = StateGraph(AgentState)

        # 添加节点
        builder.add_node(
            "inspect_video",
            self.inspect_video,
        )

        builder.add_node(
            "plan_segments",
            self.plan_segments,
        )

        builder.add_node(
            "analyze_segments",
            self.analyze_segments,
        )

        builder.add_node(
            "retrieve_guidance",
            self.retrieve_guidance,
        )

        builder.add_node(
            "write_report",
            self.write_report,
        )

        # ----------------------------------------------------
        # 添加节点之间的连接关系
        # ----------------------------------------------------

        builder.add_edge(
            START,
            "inspect_video",
        )

        builder.add_edge(
            "inspect_video",
            "plan_segments",
        )

        builder.add_edge(
            "plan_segments",
            "analyze_segments",
        )

        builder.add_edge(
            "analyze_segments",
            "retrieve_guidance",
        )

        builder.add_edge(
            "retrieve_guidance",
            "write_report",
        )

        builder.add_edge(
            "write_report",
            END,
        )

        # 编译 Graph
        return builder.compile()

    # ========================================================
    # Step 1：读取视频
    # ========================================================

    def inspect_video(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        第一阶段：

        读取视频信息，并对整个视频进行粗采样。

        这里相当于 Agent 的“第一次观察”。

        例如：
        4.5 秒视频 → 均匀抽取 8 帧
        """

        # 获取视频 metadata
        meta = video_metadata(
            state["video_path"]
        )

        # 创建临时工作目录
        work_dir = tempfile.mkdtemp(
            prefix="driving_agent_"
        )

        # 对整个视频进行粗采样
        coarse = coarse_video_frames(
            state["video_path"],
            self.config.coarse_frames,
        )

        # 保存粗采样帧
        paths = save_frames(
            coarse,
            str(
                Path(work_dir) / "coarse"
            ),
        )

        # 保存到 Agent State
        coarse_info = []

        for (idx, _), path in zip(
            coarse,
            paths,
        ):
            coarse_info.append(
                {
                    "frame_idx": idx,
                    "path": path,
                }
            )

        return {
            "metadata": meta,
            "work_dir": work_dir,
            "coarse_frames": coarse_info,
        }

    # ========================================================
    # Step 2：Planner
    # ========================================================

    def plan_segments(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        第二阶段：

        让 Qwen 根据第一次粗采样的帧，
        决定哪些时间段值得进一步检查。

        这一步体现 Agent 的“决策能力”。

        例如：

        整个视频
            ↓
        看到行人出现
            ↓
        决定：
        1.8s ~ 2.7s 值得进一步分析
        """

        # ----------------------------------------------------
        # 构建 Planner Prompt
        # ----------------------------------------------------

        prompt = PLANNER_PROMPT.format(
            max_segments=self.config.max_segments,
            duration=state["metadata"]["duration_sec"],
        )

        # ----------------------------------------------------
        # 调用视觉模型
        #
        # 注意：
        # 现在这里最终会连接 AutoDL 上的 Qwen2.5-VL。
        # ----------------------------------------------------

        result = self.client.json_chat(
            text=prompt,
            image_paths=[
                x["path"]
                for x in state["coarse_frames"]
            ],
            system_prompt=SYSTEM_PROMPT,
            max_tokens=700,
        )

        # 获取模型返回的 segments
        segments = (
            result.get("segments", [])
            if isinstance(result, dict)
            else []
        )

        duration = float(
            state["metadata"]["duration_sec"]
        )

        cleaned = []

        # ----------------------------------------------------
        # 对模型产生的时间段进行清理
        #
        # 防止模型输出：
        # - 负数时间
        # - 超出视频长度
        # - 非法数据
        # ----------------------------------------------------

        for seg in segments[
            : self.config.max_segments
        ]:

            try:
                start = max(
                    0.0,
                    min(
                        duration,
                        float(seg["start_sec"]),
                    ),
                )

                end = max(
                    start,
                    min(
                        duration,
                        float(seg["end_sec"]),
                    ),
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            # 至少保留 0.5 秒
            if end - start < 0.5:
                end = min(
                    duration,
                    start + 0.5,
                )

            cleaned.append(
                {
                    "start_sec": round(
                        start,
                        3,
                    ),
                    "end_sec": round(
                        end,
                        3,
                    ),
                    "priority": str(
                        seg.get(
                            "priority",
                            "medium",
                        )
                    ),
                    "reason": str(
                        seg.get(
                            "reason",
                            "",
                        )
                    ),
                    "topics": (
                        list(
                            seg.get(
                                "topics",
                                [],
                            )
                        )
                        if isinstance(
                            seg.get(
                                "topics",
                                [],
                            ),
                            list,
                        )
                        else []
                    ),
                }
            )

        # ----------------------------------------------------
        # 如果 Planner 没有找到任何时间段
        # 使用一个安全 fallback
        # ----------------------------------------------------

        if not cleaned:

            mid = duration / 2.0

            half = min(
                2.0,
                max(
                    0.5,
                    duration / 6.0,
                ),
            )

            cleaned = [
                {
                    "start_sec": max(
                        0,
                        mid - half,
                    ),
                    "end_sec": min(
                        duration,
                        mid + half,
                    ),
                    "priority": "medium",
                    "reason": "planner fallback",
                    "topics": [],
                }
            ]

        return {
            "plan": cleaned
        }

    # ========================================================
    # Step 3：重点时间段分析
    # ========================================================

    def analyze_segments(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        第三阶段：

        根据 Planner 选出来的时间段，
        对这些局部区域重新抽帧并进行详细分析。

        这一步就是把实验二的 Frame Sampling
        真正接入 Agent。

        Part 2：
            研究怎样采样

        Part 3：
            Agent 决定什么时候调用采样工具
        """

        results = []

        fps = float(
            state["metadata"]["fps"]
        )

        segment_root = (
            Path(state["work_dir"])
            / "segments"
        )

        # 对每一个重点时间段分别分析
        for i, seg in enumerate(
            state["plan"]
        ):

            # -----------------------------------------------
            # 对重点时间段重新采样
            # -----------------------------------------------

            frames = segment_frames(
                video_path=state["video_path"],
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                fps=fps,
                k=self.config.segment_frames,
            )

            # 保存这些局部帧
            paths = save_frames(
                frames,
                str(
                    segment_root
                    / f"segment_{i:02d}"
                ),
            )

            # -----------------------------------------------
            # 让 Qwen 对这个时间段进行详细分析
            # -----------------------------------------------

            raw = self.client.json_chat(
                text=SEGMENT_PROMPT,
                image_paths=paths,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=900,
            )

            # 保存分析结果
            results.append(
                {
                    "segment_id": i,
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "priority": seg["priority"],
                    "planner_reason": seg["reason"],
                    "planner_topics": seg["topics"],
                    "frames": [
                        idx
                        for idx, _ in frames
                    ],
                    "analysis": raw,
                }
            )

        return {
            "segment_results": results
        }

    # ========================================================
    # Step 4：RAG
    # ========================================================

    def retrieve_guidance(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        第四阶段：

        根据前面的视觉分析结果，
        查询驾驶安全知识库。

        例如：

        视频分析：
        "pedestrian crossing"

                ↓

        RAG Query：
        "pedestrian crossing safety"

                ↓

        knowledge_base.json

                ↓

        返回：
        Vulnerable road user
        Crossing object
        Intersection safety
        """

        queries = []

        # 从每个 segment 的分析结果中提取关键词
        for result in state[
            "segment_results"
        ]:

            analysis = result.get(
                "analysis",
                {},
            )

            # Planner 给出的 topics
            topics = result.get(
                "planner_topics",
                [],
            )

            # Segment Analysis 给出的 safety_topics
            if isinstance(
                analysis,
                dict,
            ):
                topics += analysis.get(
                    "safety_topics",
                    [],
                )

            # 把 topic 组合成 query
            queries.append(
                " ".join(
                    map(
                        str,
                        topics,
                    )
                )
            )

            # 同时把 assessment 作为 query
            if isinstance(
                analysis,
                dict,
            ):
                queries.append(
                    str(
                        analysis.get(
                            "assessment",
                            "",
                        )
                    )
                )

        # ----------------------------------------------------
        # 对检索结果去重
        # ----------------------------------------------------

        docs_by_id = {}

        for query in queries:

            if not query.strip():
                continue

            docs = self.retriever.search(
                query,
                self.config.top_k_docs,
            )

            for doc in docs:
                docs_by_id[
                    doc["id"]
                ] = doc

        # 返回检索到的知识
        return {
            "retrieved_docs": list(
                docs_by_id.values()
            )[
                : self.config.top_k_docs * 2
            ]
        }

    # ========================================================
    # Step 5：最终报告
    # ========================================================

    def write_report(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        第五阶段：

        把：

        1. 视频信息
        2. Agent 选择的时间段
        3. 详细视觉分析
        4. RAG 检索结果

        全部交给 Qwen，
        生成最终安全分析报告。
        """

        # ----------------------------------------------------
        # 汇总整个 Case 的信息
        # ----------------------------------------------------

        packet = {
            "video_metadata": state[
                "metadata"
            ],
            "selected_segments": state[
                "plan"
            ],
            "segment_results": state[
                "segment_results"
            ],
            "retrieved_guidance": state[
                "retrieved_docs"
            ],
        }

        # ----------------------------------------------------
        # 最终 Prompt
        # ----------------------------------------------------

        prompt = (
            FINAL_REPORT_PROMPT
            + "\n\nCASE DATA:\n"
            + json.dumps(
                packet,
                ensure_ascii=False,
                indent=2,
            )
        )

        # ----------------------------------------------------
        # 调用 Qwen 生成最终报告
        # ----------------------------------------------------

        report = self.client.chat(
            text=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1400,
        )

        return {
            "report": report
        }

    # ========================================================
    # Agent 对外入口
    # ========================================================

    def run(
        self,
        video_path: str,
    ) -> dict[str, Any]:
        """
        运行整个 Agent。

        用户只需要给 Agent 一个视频路径：

            agent.run("test_video.mp4")

        Agent 会自动执行：

            视频信息
              ↓
            粗采样
              ↓
            Planner
              ↓
            局部重新采样
              ↓
            详细分析
              ↓
            RAG
              ↓
            最终报告
        """

        # 转换成绝对路径
        video_path = str(
            Path(video_path).resolve()
        )

        # 从初始 State 开始执行 Graph
        result = self.graph.invoke(
            {
                "video_path": video_path
            }
        )

        return dict(result)

    # ========================================================
    # 清理临时文件
    # ========================================================

    @staticmethod
    def cleanup(
        result: dict[str, Any],
    ) -> None:
        """
        删除 Agent 运行过程中产生的临时图片。

        例如：
        agent_output/
        ├── coarse/
        └── segments/

        分析完成后可以清理。
        """

        work_dir = result.get(
            "work_dir"
        )

        if work_dir:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )