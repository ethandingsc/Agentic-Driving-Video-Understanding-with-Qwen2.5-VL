from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import DrivingVideoAgent
from .config import AgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: agentic driving video analysis")
    parser.add_argument("video", help="Path to a recorded driving video")
    parser.add_argument("--output", default="experiment3_report.json", help="Output JSON path")
    args = parser.parse_args()

    agent = DrivingVideoAgent(AgentConfig())
    result = agent.run(args.video)
    out = Path(args.output)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["report"])
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main()
