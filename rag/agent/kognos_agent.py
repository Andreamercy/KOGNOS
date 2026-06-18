"""
rag/agent/kognos_agent.py

KOGNOS ReAct agent — combines the RAG query engine with kubectl tools
to answer natural language questions AND autonomously remediate anomalies.

In demo mode: uses mock LLM and returns structured canned responses.
In production: uses LlamaIndex ReActAgent with real Claude/Ollama backend.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rag.agent.tools import (
    kubectl_describe,
    kubectl_logs,
    kubectl_rollback,
    kubectl_scale,
    kubectl_get_events,
)
from rag.agent.prompts import REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEMO_MODE   = os.getenv("KOGNOS_DEMO_MODE", "false").lower() == "true"
LLM_BACKEND = os.getenv("LLM_BACKEND", "anthropic").lower()


def build_agent(llm: Any | None = None) -> Any:
    """
    Build and return a KOGNOS ReAct agent.

    Args:
        llm: Pre-built LLM instance (optional). If None, builds from env config.

    Returns:
        A ReActAgent instance (or MockAgent in demo mode).
    """
    if DEMO_MODE:
        logger.info("Building mock KOGNOS agent (demo mode)")
        return MockAgent()

    try:
        from llama_index.core.agent import ReActAgent
        from llama_index.core.tools import FunctionTool
    except ImportError:
        logger.warning("llama-index not available — using mock agent")
        return MockAgent()

    if llm is None:
        if LLM_BACKEND == "anthropic":
            from rag.llm.claude_backend import build_claude_llm
            llm = build_claude_llm()
        else:
            from rag.llm.ollama_backend import build_ollama_llm
            llm = build_ollama_llm()

    tools = [
        FunctionTool.from_defaults(
            fn=kubectl_describe,
            name="kubectl_describe",
            description="Describe a Kubernetes pod to inspect its status, conditions, and events.",
        ),
        FunctionTool.from_defaults(
            fn=kubectl_logs,
            name="kubectl_logs",
            description="Fetch the last N lines of logs from a pod container.",
        ),
        FunctionTool.from_defaults(
            fn=kubectl_rollback,
            name="kubectl_rollback",
            description="Roll back a deployment to its previous stable revision.",
        ),
        FunctionTool.from_defaults(
            fn=kubectl_scale,
            name="kubectl_scale",
            description="Scale a deployment to a specified number of replicas.",
        ),
        FunctionTool.from_defaults(
            fn=kubectl_get_events,
            name="kubectl_get_events",
            description="Get Kubernetes events for a namespace, optionally filtered by pod name.",
        ),
    ]

    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        max_iterations=6,
        system_prompt=REACT_SYSTEM_PROMPT,
    )
    logger.info("ReAct agent built with %d tools (backend=%s)", len(tools), LLM_BACKEND)
    return agent


class MockAgent:
    """Mock agent for demo and testing — returns canned multi-step reasoning."""

    def chat(self, message: str) -> "MockAgentResponse":
        reasoning, answer = _mock_reasoning(message)
        return MockAgentResponse(response=answer, reasoning=reasoning)

    def query(self, message: str) -> "MockAgentResponse":
        return self.chat(message)


class MockAgentResponse:
    def __init__(self, response: str, reasoning: list[str]) -> None:
        self.response  = response
        self.reasoning = reasoning

    def __str__(self) -> str:
        return self.response


def _mock_reasoning(question: str) -> tuple[list[str], str]:
    """Simulate multi-step ReAct reasoning for demo purposes."""
    q = question.lower()

    if "payment" in q or "500" in q:
        reasoning = [
            "Thought: The question is about payment-svc errors. Let me describe the pod first.",
            "Action: kubectl_describe('payment-svc', 'production')",
            "Observation: OOMKilling event found in pod events.",
            "Thought: OOMKill detected. Let me check recent logs to confirm.",
            "Action: kubectl_logs('payment-svc', 'production', 30)",
            "Observation: OutOfMemoryError in logs confirms memory pressure.",
            "Thought: Root cause is OOMKill. Runbook suggests scaling up replicas.",
            "Action: kubectl_scale('payment-svc', 5, 'production')",
            "Observation: [DRY RUN] Scale command queued (DRY_RUN=true).",
            "Thought: Remediation prepared. Composing answer.",
        ]
        answer = (
            "**Diagnosis**: payment-svc is experiencing OOMKill events. "
            "The JVM heap is exhausted under load, causing the pod to be killed "
            "before completing payment requests.\n\n"
            "**Evidence**: OOMKilling event in pod events; "
            "java.lang.OutOfMemoryError in logs (score: 0.91 CRITICAL).\n\n"
            "**Recommended Action**:\n```bash\n"
            "kubectl scale deployment/payment-svc --replicas=5 -n production\n```\n\n"
            "**Risk Level**: Low — adding replicas distributes load without downtime. "
            "Monitor memory usage after scaling."
        )
    elif "rollback" in q:
        reasoning = [
            "Thought: User wants to rollback. Let me identify the deployment.",
            "Action: kubectl_rollback('checkout-svc', 'production')",
            "Observation: [DRY RUN] rollout undo queued.",
        ]
        answer = (
            "**Action Queued** (dry-run):\n```bash\n"
            "kubectl rollout undo deployment/checkout-svc -n production\n```\n\n"
            "Set DRY_RUN=false to execute. This will revert to the previous revision."
        )
    else:
        reasoning = ["Thought: General cluster health question."]
        answer = (
            "I'm KOGNOS Agent. I can help you diagnose pods, fetch logs, "
            "rollback deployments, and scale services. "
            "Ask me something specific like: 'Why is payment-svc down?' "
            "or 'Roll back the checkout service.'"
        )

    return reasoning, answer
