from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


# 简单英文分词
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_-]+")


class SafetyRetriever:
    """基于关键词和 TF-IDF 的轻量级安全知识检索器。"""

    def __init__(self, kb_path: str | Path):
        # 加载知识库
        self.kb_path = Path(kb_path)
        self.docs: list[dict[str, Any]] = json.loads(
            self.kb_path.read_text(encoding="utf-8")
        )

        # 统计每个词在多少个文档中出现
        self.idf = self._build_idf()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """把文本简单切成小写 token。"""
        return [
            x.lower()
            for x in _TOKEN_RE.findall(text or "")
        ]

    def _build_idf(self) -> dict[str, float]:
        """计算 IDF，让常见词权重降低。"""

        df: dict[str, int] = {}

        for doc in self.docs:
            # title + content + tags 一起参与统计
            text = " ".join(
                [
                    str(doc.get("title", "")),
                    str(doc.get("content", "")),
                    " ".join(
                        map(str, doc.get("tags", []))
                    ),
                ]
            )

            # 一个词在同一个文档里只算一次
            seen = set(self._tokens(text))

            for token in seen:
                df[token] = df.get(token, 0) + 1

        n = max(1, len(self.docs))

        return {
            token: math.log(
                (1 + n) / (1 + freq)
            ) + 1.0
            for token, freq in df.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        根据 query 检索最相关的安全知识。

        检索优先级：
        tags > title > content

        这样 pedestrian 查询更容易找到 pedestrian
        相关知识，而不是被其他文档中的常见词带偏。
        """

        q_tokens = self._tokens(query)

        if not q_tokens:
            return []

        scored: list[
            tuple[float, dict[str, Any]]
        ] = []

        for doc in self.docs:

            # 分别处理三个字段
            title_tokens = self._tokens(
                str(doc.get("title", ""))
            )

            content_tokens = self._tokens(
                str(doc.get("content", ""))
            )

            tag_tokens = self._tokens(
                " ".join(
                    map(
                        str,
                        doc.get("tags", []),
                    )
                )
            )

            # 统计每个字段中 token 出现次数
            title_counts: dict[str, int] = {}
            content_counts: dict[str, int] = {}
            tag_counts: dict[str, int] = {}

            for token in title_tokens:
                title_counts[token] = (
                    title_counts.get(token, 0) + 1
                )

            for token in content_tokens:
                content_counts[token] = (
                    content_counts.get(token, 0) + 1
                )

            for token in tag_tokens:
                tag_counts[token] = (
                    tag_counts.get(token, 0) + 1
                )

            score = 0.0

            # ---------------------------------------------
            # 不同字段使用不同权重
            # ---------------------------------------------
            #
            # tags    = 3.0
            # title   = 2.0
            # content = 1.0
            #
            # 场景标签比普通正文更加重要。
            # ---------------------------------------------

            for token in q_tokens:

                idf = self.idf.get(
                    token,
                    1.0,
                )

                # tags 匹配权重最高
                if token in tag_counts:
                    score += (
                        3.0
                        * (
                            1.0
                            + math.log(
                                tag_counts[token]
                            )
                        )
                        * idf
                    )

                # title 匹配次之
                if token in title_counts:
                    score += (
                        2.0
                        * (
                            1.0
                            + math.log(
                                title_counts[token]
                            )
                        )
                        * idf
                    )

                # content 匹配权重最低
                if token in content_counts:
                    score += (
                        1.0
                        * (
                            1.0
                            + math.log(
                                content_counts[token]
                            )
                        )
                        * idf
                    )

            # 只保留有匹配的文档
            if score > 0:
                scored.append(
                    (
                        score,
                        doc,
                    )
                )

        # 按相关性从高到低排序
        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # 返回 Top-K
        return [
            dict(
                doc,
                score=round(score, 4),
            )
            for score, doc in scored[:top_k]
        ]