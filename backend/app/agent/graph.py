"""Official LangGraph workflow definition and compilation for SmartShopper."""

from __future__ import annotations

from typing import Any
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agent.nodes import (
    adjust_search_node,
    evaluate_results_node,
    product_comparison_node,
    product_search_node,
    request_planning_node,
    response_metadata_node,
    safe_fallback_node,
    shopper_profile_node,
    simulate_preferences_node,
)
from app.agent.routing import (
    route_after_evaluation,
    route_after_planning,
)
from app.agent.state import (
    AgentState,
    ShopperGraphState,
    agent_state_to_graph_state,
)
from app.agent.tool_registry import ToolRegistry
from app.services.groq_client import GroqClient


def build_shopper_graph() -> Any:
    """Builds and compiles the SmartShopper LangGraph StateGraph workflow."""
    workflow = StateGraph(ShopperGraphState)

    # Register workflow nodes
    workflow.add_node("planning", request_planning_node)
    workflow.add_node("product_search", product_search_node)
    workflow.add_node("product_comparison", product_comparison_node)
    workflow.add_node("get_shopper_profile", shopper_profile_node)
    workflow.add_node("simulate_preferences", simulate_preferences_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
    workflow.add_node("adjust_search", adjust_search_node)
    workflow.add_node("safe_fallback", safe_fallback_node)
    workflow.add_node("response_metadata", response_metadata_node)

    # Workflow entry point
    workflow.add_edge(START, "planning")

    # Routing from planning node
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "product_search": "product_search",
            "product_comparison": "product_comparison",
            "get_shopper_profile": "get_shopper_profile",
            "simulate_preferences": "simulate_preferences",
            "evaluate_results": "evaluate_results",
            "safe_fallback": "safe_fallback",
            "response_metadata": "response_metadata",
        },
    )

    # Tool execution nodes route to evaluation
    workflow.add_edge("product_search", "evaluate_results")
    workflow.add_edge("product_comparison", "evaluate_results")
    workflow.add_edge("get_shopper_profile", "evaluate_results")
    workflow.add_edge("simulate_preferences", "evaluate_results")
    workflow.add_edge("safe_fallback", "evaluate_results")

    # Routing from evaluation node
    workflow.add_conditional_edges(
        "evaluate_results",
        route_after_evaluation,
        {
            "planning": "planning",
            "product_comparison": "product_comparison",
            "adjust_search": "adjust_search",
            "response_metadata": "response_metadata",
        },
    )

    # Retry loop: adjust_search retries product_search
    workflow.add_edge("adjust_search", "product_search")

    # Termination
    workflow.add_edge("response_metadata", END)

    return workflow.compile()


# Module-level compiled graph singleton
_compiled_shopper_graph = build_shopper_graph()


def get_compiled_graph() -> Any:
    """Returns the compiled LangGraph workflow instance."""
    return _compiled_shopper_graph


def run_shopper_graph(
    state: AgentState,
    db: Session,
    client: GroqClient,
    registry: ToolRegistry,
) -> ShopperGraphState:
    """Executes the request-scoped LangGraph workflow and returns the final graph state."""
    graph = get_compiled_graph()
    initial_graph_state = agent_state_to_graph_state(state)
    config = {
        "configurable": {
            "db": db,
            "client": client,
            "registry": registry,
        }
    }
    final_output = graph.invoke(initial_graph_state, config=config)
    return final_output
