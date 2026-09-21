from experiment3_agent.agent import DrivingVideoAgent


# 测试视频
video_path = r"C:\MyZbook\Research_projects\MLLM\行车agent测试数据\test1行人过马路.mp4"


# 创建 Agent
agent = DrivingVideoAgent()

# 运行完整 Agent 流程
result = agent.run(video_path)


# ============================================================
# 1. Planner 选择的重点时间段
# ============================================================

print("=== PLAN ===")

for segment in result["plan"]:
    print(segment)


# ============================================================
# 2. 重点时间段的视觉分析
# ============================================================

print("\n=== SEGMENT RESULTS ===")

for result_item in result["segment_results"]:
    print(
        f"Segment {result_item['segment_id']}: "
        f"{result_item['start_sec']}s - "
        f"{result_item['end_sec']}s"
    )

    print(result_item["analysis"])
    print()


# ============================================================
# 3. RAG 检索结果
# ============================================================

print("\n=== RAG ===")

for doc in result["retrieved_docs"]:
    print(
        f"[{doc['score']}] "
        f"{doc['title']}"
    )


# ============================================================
# 4. 最终报告
# ============================================================

print("\n=== REPORT ===")
print(result["report"])


# ============================================================
# 5. 清理临时图片
# ============================================================

agent.cleanup(result)