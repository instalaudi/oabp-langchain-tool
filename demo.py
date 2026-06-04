"""Smoke test for oabp_tool.

Runs the three required primitives against the AIGEN reference server
to confirm the tool wires up correctly. Does not submit any work.

Usage:
    python demo.py
"""
from __future__ import annotations

import json

from oabp_tool import (
    OABCPCheckReputationTool,
    OABPClient,
    OABPListMissionsTool,
    OABCPSubmitSolutionTool,
)


def main() -> None:
    client = OABPClient(agent_id="opencode-langchain-agent-demo")

    list_tool = OABPListMissionsTool(client=client)
    rep_tool = OABCPCheckReputationTool(client=client)

    print("=== list_open_missions() ===")
    print(list_tool._run(limit=5))

    print("\n=== check_agent_reputation('aigen-autopilot') ===")
    print(rep_tool._run(agent_id="aigen-autopilot"))

    submit_tool = OABCPSubmitSolutionTool(client=client)
    print("\n=== submit_solution(...) dry-run schema (no actual POST) ===")
    print(json.dumps({
        "mission_id": "mis_334ad09eccaa",
        "proof_url": "https://github.com/EXAMPLE/EXAMPLE",
        "metadata_json": None,
    }, indent=2))
    print("submit tool wired up, ready to call submit_solution() with real args.")


if __name__ == "__main__":
    main()
