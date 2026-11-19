"""Request-scoped state for a bounded assistant agent run and LangGraph workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict
from uuid import UUID, uuid4

from app.agent.schemas import EvaluationResult, ToolSelection
from app.schemas.assistant import AssistantChatResponse, AssistantIntent

ToolSelectionSource = Literal[
    "planner",
    "policy",
    "fallback",
    "evaluation",
]


@dataclass
class ToolExecutionRecord:
    tool_name: str
    arguments: dict[str, object]
    result: object | None = None
    error_code: str | None = None
    executed: bool = True
    selection_source: ToolSelectionSource = "planner"


@dataclass
class AgentState:
    objective: str
    user_id: int
    conversation_context: AssistantIntent | None
    constraints: AssistantIntent
    previously_shown_product_ids: list[str]
    exclude_previous_products: bool
    request_limit: int
    comparison_requested: bool = False
    fallback_search_allowed: bool = True
    execution_id: UUID = field(default_factory=uuid4)
    current_plan: ToolSelection | None = None
    planning_failed: bool = False
    completed_steps: list[str] = field(default_factory=list)
    tool_results: list[ToolExecutionRecord] = field(default_factory=list)
    remaining_execution_budget: int = 5
    tool_execution_count: int = 0
    latest_evaluation: EvaluationResult | None = None
    execution_status: Literal[
        "completed",
        "partial",
        "fallback",
        "unavailable",
    ] = "completed"
    final_response: AssistantChatResponse | None = None

    @classmethod
    def create(
        cls,
        *,
        objective: str,
        user_id: int,
        conversation_context: AssistantIntent | None,
        constraints: AssistantIntent,
        previously_shown_product_ids: list[str],
        exclude_previous_products: bool,
        request_limit: int,
        comparison_requested: bool = False,
        fallback_search_allowed: bool = True,
    ) -> AgentState:
        return cls(
            objective=objective,
            user_id=user_id,
            conversation_context=conversation_context,
            constraints=constraints,
            previously_shown_product_ids=list(previously_shown_product_ids),
            exclude_previous_products=exclude_previous_products,
            request_limit=request_limit,
            comparison_requested=comparison_requested,
            fallback_search_allowed=fallback_search_allowed,
        )


class ShopperGraphState(TypedDict, total=False):
    """Isolated, request-scoped TypedDict state used by the LangGraph workflow."""

    objective: str
    user_id: int
    conversation_context: AssistantIntent | None
    constraints: AssistantIntent
    previously_shown_product_ids: list[str]
    exclude_previous_products: bool
    request_limit: int
    comparison_requested: bool
    fallback_search_allowed: bool
    execution_id: UUID
    current_plan: ToolSelection | None
    planning_failed: bool
    completed_steps: list[str]
    tool_results: list[ToolExecutionRecord]
    remaining_execution_budget: int
    tool_execution_count: int
    planning_iteration_count: int
    latest_evaluation: EvaluationResult | None
    execution_status: Literal["completed", "partial", "fallback", "unavailable"]
    seen_calls: list[str]
    next_action: str | None


def agent_state_to_graph_state(state: AgentState) -> ShopperGraphState:
    """Explicitly map an existing AgentState into a LangGraph ShopperGraphState."""
    return {
        "objective": state.objective,
        "user_id": state.user_id,
        "conversation_context": state.conversation_context,
        "constraints": state.constraints,
        "previously_shown_product_ids": list(state.previously_shown_product_ids),
        "exclude_previous_products": state.exclude_previous_products,
        "request_limit": state.request_limit,
        "comparison_requested": state.comparison_requested,
        "fallback_search_allowed": state.fallback_search_allowed,
        "execution_id": state.execution_id,
        "current_plan": state.current_plan,
        "planning_failed": state.planning_failed,
        "completed_steps": list(state.completed_steps),
        "tool_results": list(state.tool_results),
        "remaining_execution_budget": state.remaining_execution_budget,
        "tool_execution_count": state.tool_execution_count,
        "planning_iteration_count": 0,
        "latest_evaluation": state.latest_evaluation,
        "execution_status": state.execution_status,
        "seen_calls": [],
        "next_action": None,
    }


def graph_state_to_agent_state(graph_state: ShopperGraphState) -> AgentState:
    """Explicitly map a LangGraph ShopperGraphState back into an AgentState."""
    return AgentState(
        objective=graph_state["objective"],
        user_id=graph_state["user_id"],
        conversation_context=graph_state.get("conversation_context"),
        constraints=graph_state["constraints"],
        previously_shown_product_ids=list(
            graph_state.get("previously_shown_product_ids", [])
        ),
        exclude_previous_products=graph_state.get("exclude_previous_products", False),
        request_limit=graph_state.get("request_limit", 10),
        comparison_requested=graph_state.get("comparison_requested", False),
        fallback_search_allowed=graph_state.get("fallback_search_allowed", True),
        execution_id=graph_state.get("execution_id") or uuid4(),
        current_plan=graph_state.get("current_plan"),
        planning_failed=graph_state.get("planning_failed", False),
        completed_steps=list(graph_state.get("completed_steps", [])),
        tool_results=list(graph_state.get("tool_results", [])),
        remaining_execution_budget=graph_state.get("remaining_execution_budget", 5),
        tool_execution_count=graph_state.get("tool_execution_count", 0),
        latest_evaluation=graph_state.get("latest_evaluation"),
        execution_status=graph_state.get("execution_status", "completed"),
    )
