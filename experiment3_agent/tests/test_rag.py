from pathlib import Path

from experiment3_agent.rag import SafetyRetriever


def test_retrieval_returns_pedestrian_guidance():
    kb = Path(__file__).parents[1] / "experiment3_agent" / "knowledge_base.json"
    retriever = SafetyRetriever(kb)
    docs = retriever.search("pedestrian crosswalk yield")
    assert docs
    assert docs[0]["id"] == "pedestrian_crosswalk"
