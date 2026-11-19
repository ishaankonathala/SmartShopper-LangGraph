"""Comprehensive integration and node tests for the LangGraph SmartShopper workflow."""

from __future__ import annotations

import json
from typing import Any
import pytest
from sqlalchemy.orm import Session

from app.agent.graph import build_shopper_graph, get_compiled_graph, run_shopper_graph
from app.agent.orchestrator import (
    MAX_AGENT_SELECTION_ITERATIONS,
    MAX_AGENT_TOOL_EXECUTIONS,
    AgentOrchestrator,
)
from app.agent.routing import (
    _comparison_selection,
    route_after_evaluation,
    route_after_planning,
)
from app.agent.schemas import (
    EvaluationResult,
    ProductComparisonResult,
    ShopperProfileResult,
    ToolSelection,
)
from app.agent.state import (
    AgentState,
    ShopperGraphState,
    ToolExecutionRecord,
    agent_state_to_graph_state,
    graph_state_to_agent_state,
)
from app.agent.tools import create_tool_registry
from app.models.product import Product
from app.schemas.assistant import AssistantIntent
from app.schemas.profile import ProfileCreate
from app.schemas.recommendation import RecommendationResponse
from app.services import profile as profile_service
from app.services.groq_client import GroqClient, GroqNetworkError
from tests.test_agent_architecture import (
    SequencePlannerClient,
    _add_amazon_product,
    _add_curated_product,
    _create_profile,
    _state,
)


def test_compiled_graph_structure():
    """Verify that the compiled LangGraph workflow registers all expected nodes."""
    graph = get_compiled_graph()
    assert graph is not None
    # Verify graph node names
    expected_nodes = {
        "planning",
        "product_search",
        "product_comparison",
        "get_shopper_profile",
        "simulate_preferences",
        "evaluate_results",
        "adjust_search",
        "safe_fallback",
        "response_metadata",
    }
    registered_nodes = set(graph.nodes.keys())
    assert expected_nodes.issubset(registered_nodes)


def test_langgraph_state_conversion_roundtrip():
    """Verify lossless conversion between AgentState and ShopperGraphState."""
    original_state = _state(1, comparison=True, limit=7)
    original_state.completed_steps.append("Initial step")

    graph_state = agent_state_to_graph_state(original_state)
    assert graph_state["objective"] == original_state.objective
    assert graph_state["user_id"] == original_state.user_id
    assert graph_state["request_limit"] == 7
    assert graph_state["comparison_requested"] is True
    assert graph_state["completed_steps"] == ["Initial step"]

    restored_state = graph_state_to_agent_state(graph_state)
    assert restored_state.objective == original_state.objective
    assert restored_state.user_id == original_state.user_id
    assert restored_state.request_limit == 7
    assert restored_state.comparison_requested is True
    assert restored_state.completed_steps == ["Initial step"]


def test_langgraph_e2e_search_and_evaluation(db_session: Session):
    """Integration test: execute compiled graph for product search and evaluation."""
    user_id = _create_profile(db_session)
    _add_curated_product(db_session, "LG-P1", price=3000)
    _add_curated_product(db_session, "LG-P2", price=4000)

    planner_client = SequencePlannerClient(
        [
            {"tool_name": "product_search", "arguments": {"limit": 5}},
        ]
    )
    state = _state(user_id, comparison=False, limit=5)
    orchestrator = AgentOrchestrator(planner_client)
    result = orchestrator.run(state, db_session)

    assert result.recommendation_response is not None
    assert len(result.recommendation_response.products) >= 2
    assert result.state.execution_status == "completed"
    assert result.evaluation_result is not None
    assert result.evaluation_result.satisfied is True
    assert result.evaluation_result.outcome == "eligible_results"

    # Verify activity events generated from the LangGraph run
    metadata = result.metadata
    assert metadata.status == "completed"
    assert "product_search" in metadata.tools_executed
    assert "evaluate_results" in metadata.tools_executed
    kinds = [event.kind for event in metadata.events]
    assert kinds[0] == "understanding"
    assert kinds[1] == "planning"
    assert kinds[-1] == "completion"


def test_langgraph_e2e_comparison_routing(db_session: Session):
    """Integration test: product search followed by automatic comparison in graph."""
    user_id = _create_profile(db_session)
    _add_curated_product(db_session, "COMP-1", price=3200)
    _add_curated_product(db_session, "COMP-2", price=4500)

    planner_client = SequencePlannerClient(
        [
            {"tool_name": "product_search", "arguments": {"limit": 5}},
        ]
    )
    state = _state(user_id, comparison=True, limit=5)
    result = AgentOrchestrator(planner_client).run(state, db_session)

    assert result.comparison_result is not None
    assert len(result.comparison_result.products) == 2
    assert result.state.execution_status == "completed"
    assert result.evaluation_result is not None
    assert result.evaluation_result.outcome == "comparison_complete"


def test_langgraph_e2e_shopper_profile_node(db_session: Session):
    """Integration test: explicit shopper profile query execution."""
    user_id = _create_profile(
        db_session,
        preferred_brands=["Sony", "Bose"],
    )
    planner_client = SequencePlannerClient(
        [
            {"tool_name": "get_shopper_profile", "arguments": {}},
        ]
    )
    state = _state(
        user_id,
        intent=AssistantIntent(category=None),
        limit=5,
    )
    result = AgentOrchestrator(planner_client).run(state, db_session)

    assert result.profile_result is not None
    assert result.profile_result.preferred_brands == ["Sony", "Bose"]
    assert result.evaluation_result is not None
    assert result.evaluation_result.outcome == "profile_retrieved"

def test_shopper_profile_request_bypasses_planner(db_session: Session):
    """Explicit profile requests should not depend on Groq planning."""
    user_id = _create_profile(
        db_session,
        preferred_brands=["Sony", "Bose"],
    )

    planner_client = SequencePlannerClient([])
    state = _state(
        user_id,
        intent=AssistantIntent(category=None),
        limit=5,
    )
    state.objective = "Show my shopping profile"

    result = AgentOrchestrator(planner_client).run(state, db_session)

    assert result.profile_result is not None
    assert result.profile_result.preferred_brands == ["Sony", "Bose"]
    assert planner_client.calls == []


def test_langgraph_e2e_simulation_node(db_session: Session):
    """Integration test: hypothetical preference simulation execution."""
    user_id = _create_profile(db_session, max_budget=4000)
    _add_curated_product(db_session, "SIM-1", price=7500)

    planner_client = SequencePlannerClient(
        [
            {
                "tool_name": "simulate_preferences",
                "arguments": {"max_budget_override": 8000.0},
            },
        ]
    )
    state = _state(
        user_id,
        intent=AssistantIntent(
            category="Headphones",
            max_budget_override=8000.0,
            connectivity_requirement="wireless",
        ),
        limit=5,
    )
    result = AgentOrchestrator(planner_client).run(state, db_session)

    assert result.simulation_result is not None
    assert len(result.simulation_result.recommendations.products) >= 1
    assert result.evaluation_result is not None
    assert result.evaluation_result.outcome == "simulated_results"


def test_langgraph_groq_failure_triggers_safe_fallback(db_session: Session):
    """Integration test: when Groq planner fails with network error, fallback runs."""
    user_id = _create_profile(db_session)
    _add_curated_product(db_session, "FB-1", price=2500)

    class FailingGroqClient(GroqClient):
        def complete_json(self, messages: list[dict[str, str]]) -> str:
            raise GroqNetworkError("Simulated Groq network timeout.")

    state = _state(user_id, limit=5, fallback_search_allowed=True)
    result = AgentOrchestrator(FailingGroqClient()).run(state, db_session)

    assert result.state.planning_failed is True
    assert result.state.execution_status == "fallback"
    assert result.recommendation_response is not None
    assert len(result.recommendation_response.products) >= 1
    assert "product_search" in result.metadata.tools_executed


def test_langgraph_retry_and_iteration_limit(db_session: Session):
    """Verify that repeated planning iterations terminate at limit (3 iterations)."""
    user_id = _create_profile(db_session)
    planner_client = SequencePlannerClient(
        [
            {"tool_name": "get_shopper_profile", "arguments": {}},
            {"tool_name": "get_shopper_profile", "arguments": {}},
            {"tool_name": "get_shopper_profile", "arguments": {}},
            {"tool_name": "get_shopper_profile", "arguments": {}},
        ]
    )
    state = _state(
        user_id,
        intent=AssistantIntent(
            category="Headphones",
            max_budget_override=8000.0,
            connectivity_requirement="wireless",
        ),
        limit=5,
    )
    result = AgentOrchestrator(planner_client).run(state, db_session)

    # Max planning iterations is 3; repeated call is prevented
    assert result.state.execution_status == "partial"
    assert "Repeated tool call prevented." in result.state.completed_steps


def test_langgraph_empty_product_results(db_session: Session):
    """Verify empty product results produce no_matches evaluation without error."""
    user_id = _create_profile(db_session)
    planner_client = SequencePlannerClient(
        [
            {"tool_name": "product_search", "arguments": {"limit": 5}},
        ]
    )
    # Filter for impossible budget
    state = _state(
        user_id,
        intent=AssistantIntent(category="Headphones", max_budget_override=10),
    )
    result = AgentOrchestrator(planner_client).run(state, db_session)

    assert result.evaluation_result is not None
    assert result.evaluation_result.outcome == "no_matches"
    assert result.evaluation_result.satisfied is True
    assert result.recommendation_response is not None
    assert len(result.recommendation_response.products) == 0


def test_langgraph_invalid_tool_arguments_rejected(db_session: Session):
    """Verify invalid tool arguments are cleanly rejected and recorded."""
    user_id = _create_profile(db_session)
    planner_client = SequencePlannerClient(
        [
            {
                "tool_name": "product_search",
                "arguments": {"limit": 9999},  # Exceeds max 25
            },
        ]
    )
    state = _state(
        user_id,
        intent=AssistantIntent(
            category="Headphones",
            max_budget_override=8000.0,
            connectivity_requirement="wireless",
        ),
        limit=5,
    )
    result = AgentOrchestrator(planner_client).run(state, db_session)

    assert any(
        record.error_code == "invalid_arguments"
        for record in result.state.tool_results
    )
    assert "Invalid tool arguments were rejected." in result.state.completed_steps


def test_langgraph_routing_functions():
    """Unit tests for conditional routing functions."""
    # route_after_planning with valid plan
    state_search: ShopperGraphState = {
        "current_plan": ToolSelection(
            tool_name="product_search", arguments={"limit": 5}
        ),
        "next_action": "product_search",
    }
    assert route_after_planning(state_search) == "product_search"

    # route_after_planning with fallback
    state_fallback: ShopperGraphState = {
        "planning_failed": True,
        "fallback_search_allowed": True,
        "next_action": "safe_fallback",
    }
    assert route_after_planning(state_fallback) == "safe_fallback"

    # route_after_evaluation with satisfied result
    state_eval_satisfied: ShopperGraphState = {
        "latest_evaluation": EvaluationResult(
            satisfied=True,
            outcome="eligible_results",
            eligible_product_count=3,
        ),
        "comparison_requested": False,
        "remaining_execution_budget": 3,
    }
    assert route_after_evaluation(state_eval_satisfied) == "response_metadata"

    # route_after_evaluation with constraint violation and budget for retry
    state_eval_violation: ShopperGraphState = {
        "latest_evaluation": EvaluationResult(
            satisfied=False,
            outcome="constraint_violation",
            issues=["Budget exceeded"],
        ),
        "remaining_execution_budget": 3,
        "planning_iteration_count": 1,
    }
    assert route_after_evaluation(state_eval_violation) == "adjust_search"
