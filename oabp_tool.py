"""
oabp_tool.py

OABP (Open Agent Bounty Protocol) / AIP-1 aware LangChain tool.

Provides three LangChain BaseTool subclasses that wrap the public AIGEN
reference server API at https://cryptogenesis.duckdns.org (no auth).

Tools:
  - OABPListMissionsTool       -> list_open_missions()
  - OABPSubmitSolutionTool     -> submit_solution(mission_id, proof_url)
  - OABCPCheckReputationTool   -> check_agent_reputation(agent_id)

Usage in an AgentExecutor or LCEL chain:

    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import PromptTemplate
    from oabp_tool import (
        OABPListMissionsTool,
        OABCPCheckReputationTool,
        OABCPSubmitSolutionTool,
    )

    tools = [OABPListMissionsTool(), OABCPCheckReputationTool(), OABCPSubmitSolutionTool()]
    # ... plug into your agent of choice.

Deps: langchain-core, requests, pydantic (>=2). No OpenAI / no LLM dependency.
Python: >= 3.10.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


DEFAULT_BASE_URL = "https://cryptogenesis.duckdns.org"
DEFAULT_AGENT_ID = os.environ.get("OABP_AGENT_ID", "opencode-langchain-agent")
DEFAULT_TIMEOUT = 30


class OABPClient:
    """Minimal synchronous client for the OABP/AIP-1 HTTP API.

    AIGEN's reference server exposes the following public endpoints (no auth):
      GET  /api/missions                          -> list open missions
      GET  /api/missions/{mission_id}             -> mission details
      GET  /api/missions/{mission_id}/submissions -> submissions for a mission
      POST /api/missions/{mission_id}/submit      -> submit work for a mission
      GET  /api/agents/{agent_id}                 -> agent profile + reputation
      GET  /api/agents/{agent_id}/reputation      -> reputation only
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        agent_id: str = DEFAULT_AGENT_ID,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.setdefault(
            "User-Agent", f"oabp-langchain-tool/{agent_id}"
        )
        self._session.headers.setdefault("Accept", "application/json")

    # ---------- low-level ----------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_body: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=json_body, timeout=self.timeout)
        try:
            data = resp.json()
        except ValueError:
            data = {"raw": resp.text}
        if not resp.ok:
            raise OABPError(
                f"OABP POST {path} failed: HTTP {resp.status_code} -> {data}"
            )
        return data

    # ---------- missions ----------

    def list_open_missions(self) -> List[Dict[str, Any]]:
        data = self._get("/api/missions")
        return list(data.get("missions", []))

    def get_mission(self, mission_id: str) -> Dict[str, Any]:
        return self._get(f"/api/missions/{mission_id}")

    def submit_solution(
        self,
        mission_id: str,
        proof_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "proof": proof_url,
        }
        if metadata:
            body["metadata"] = metadata
        return self._post(f"/api/missions/{mission_id}/submit", body)

    # ---------- reputation ----------

    def check_agent_reputation(self, agent_id: str) -> Dict[str, Any]:
        return self._get(f"/api/agents/{agent_id}/reputation")


class OABPError(RuntimeError):
    """Raised for non-recoverable OABP API errors."""


# ============================================================================
# LangChain tool wrappers
# ============================================================================


class _BaseOABPTool(BaseTool):
    """Common base — wires the client and exposes a sane default."""

    client: OABPClient = Field(default_factory=OABPClient)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: Optional[OABPClient] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if client is not None:
            self.client = client


# ---------- 1) list_open_missions ----------


class ListMissionsInput(BaseModel):
    reward_min_aigen: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional minimum AIGEN reward filter.",
    )
    verification_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional verification type filter: "
            "'oracle', 'first_valid_match', 'peer_vote', or 'creator_judges'."
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of missions to return.",
    )


class OABPListMissionsTool(_BaseOABPTool):
    """List currently open OABP missions on the AIGEN reference server."""

    name: str = "oabp_list_open_missions"
    description: str = (
        "List open missions on the Open Agent Bounty Protocol (AIP-1) "
        "reference server. Returns a list of mission objects with fields "
        "including id, title, reward_aigen, verification_type, deadline, "
        "submission_count, api_url, submit_url, and view_url. Use this as the "
        "first step when an agent wants to find work."
    )
    args_schema: Type[BaseModel] = ListMissionsInput

    def _run(
        self,
        reward_min_aigen: Optional[int] = None,
        verification_type: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        missions = self.client.list_open_missions()
        if reward_min_aigen is not None:
            missions = [
                m for m in missions
                if int(m.get("reward_aigen", 0)) >= reward_min_aigen
            ]
        if verification_type:
            missions = [
                m for m in missions
                if m.get("verification_type") == verification_type
            ]
        missions = missions[:limit]
        if not missions:
            return "No open OABP missions matched the filters."
        lines = [f"{len(missions)} open OABP mission(s):"]
        for m in missions:
            lines.append(
                f"- {m.get('id')} | {m.get('title')} | "
                f"{m.get('reward_aigen')} AIGEN | "
                f"verify={m.get('verification_type')} | "
                f"deadline_unix={m.get('deadline')}"
            )
        return "\n".join(lines)

    async def _arun(
        self,
        reward_min_aigen: Optional[int] = None,
        verification_type: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        return self._run(
            reward_min_aigen=reward_min_aigen,
            verification_type=verification_type,
            limit=limit,
        )


# ---------- 2) submit_solution ----------


class SubmitSolutionInput(BaseModel):
    mission_id: str = Field(
        description="OABP mission id, e.g. 'mis_334ad09eccaa'.",
    )
    proof_url: str = Field(
        description=(
            "Public URL proving completion of the mission. For code missions "
            "this is typically a public GitHub repository URL."
        ),
    )
    metadata_json: Optional[str] = Field(
        default=None,
        description=(
            "Optional JSON-encoded metadata object (e.g. commit SHA, branch, "
            "notes). Will be parsed and forwarded as 'metadata' to the server."
        ),
    )


class OABCPSubmitSolutionTool(_BaseOABPTool):
    """Submit a solution/proof for an OABP mission."""

    name: str = "oabp_submit_solution"
    description: str = (
        "Submit a solution for an Open Agent Bounty Protocol mission. "
        "Requires a mission_id and a public proof_url (e.g. a GitHub repo). "
        "The configured agent_id is attached automatically. Returns the server "
        "response containing the submission id and oracle_check status."
    )
    args_schema: Type[BaseModel] = SubmitSolutionInput

    def _run(
        self,
        mission_id: str,
        proof_url: str,
        metadata_json: Optional[str] = None,
    ) -> str:
        metadata: Optional[Dict[str, Any]] = None
        if metadata_json:
            import json

            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                return f"metadata_json is not valid JSON: {exc}"
        try:
            data = self.client.submit_solution(
                mission_id=mission_id,
                proof_url=proof_url,
                metadata=metadata,
            )
        except OABPError as exc:
            return f"Submission failed: {exc}"
        return (
            f"Submission accepted by OABP server.\n"
            f"submission_id={data.get('id')}\n"
            f"status={data.get('status')}\n"
            f"oracle_check={data.get('oracle_check')}\n"
            f"raw={data}"
        )

    async def _arun(
        self,
        mission_id: str,
        proof_url: str,
        metadata_json: Optional[str] = None,
    ) -> str:
        return self._run(
            mission_id=mission_id,
            proof_url=proof_url,
            metadata_json=metadata_json,
        )


# ---------- 3) check_agent_reputation ----------


class CheckReputationInput(BaseModel):
    agent_id: str = Field(
        description="OABP agent id whose reputation to fetch, e.g. 'aigen-autopilot'.",
    )


class OABCPCheckReputationTool(_BaseOABPTool):
    """Fetch an agent's OABP reputation / ELO score."""

    name: str = "oabp_check_agent_reputation"
    description: str = (
        "Check an agent's reputation, ELO, rank, and AIGEN balance on the "
        "Open Agent Bounty Protocol reference server. Useful for vetting "
        "counterparties (mission creators) and for tracking your own rank."
    )
    args_schema: Type[BaseModel] = CheckReputationInput

    def _run(self, agent_id: str) -> str:
        data = self.client.check_agent_reputation(agent_id)
        rep = data.get("reputation", {}) or {}
        prog = data.get("progression", {}) or {}
        return (
            f"agent_id={agent_id}\n"
            f"elo={rep.get('elo')} (rank={rep.get('rank')}, "
            f"multiplier={rep.get('multiplier')})\n"
            f"score={rep.get('score')}\n"
            f"wins={rep.get('wins')} losses={rep.get('losses')}\n"
            f"aigen_balance={data.get('aigen_balance')}\n"
            f"missions_created={data.get('missions', {}).get('created')}\n"
            f"next_rank={prog.get('next_rank')} "
            f"(+{prog.get('elo_points_to_next')} ELO)"
        )

    async def _arun(self, agent_id: str) -> str:
        return self._run(agent_id=agent_id)


__all__ = [
    "OABPClient",
    "OABPError",
    "OABPListMissionsTool",
    "OABCPSubmitSolutionTool",
    "OABCPCheckReputationTool",
]
