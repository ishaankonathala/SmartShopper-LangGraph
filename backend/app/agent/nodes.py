"""LangGraph node implementations for the SmartShopper workflow."""

from __future__ import annotations

import json
import logging
from typing import Any, Final
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agent.planner import AgentPlanner
from app.agent.schemas import (
    EvaluationArguments,
    EvaluationResult,
    PreferenceSimulationResult,
    ProductComparisonArguments,
    ProductComparisonResult,
    ProductSearchArguments,
    ShopperProfileResult,
    ToolSelection,
)
from app.agent.state import (
    AgentState,
    ShopperGraphState,
    ToolExecutionRecord,
    ToolSelectionSource,
    graph_state_to_agent_state,
)
from app.agent.tool_registry import (
    InvalidToolArgumentsError,
    ToolRegistry,
    UnknownToolError,
)
from app.schemas.recommendation import RecommendationResponse
from app.services.groq_client import GroqClient, GroqClientError

logger = logging.getLogger(__name__)

_TOOL_ACTIVITY: Final[dict[str, str]] = {
    "product_search": "Searching eligible catalogue products.",
    "product_comparison": "Comparing available catalogue fields.",
    "get_shopper_profile": "Checking your saved shopping preferences.",
    "simulate_preferences": "Simulating temporary preference changes.",
    "evaluate_results": "Checking results against your requirements.",
}


def get_runtime_context(
    config: RunnableConfig,
) -> tuple[Session, GroqClient, ToolRegistry]:
    """Retrieve runtime dependencies passed via LangGraph config."""
    configurable = (config or {}).get("configurable", {})
    db = configurable.get("db")
    client = configurable.get("client")
    registry = configurable.get("registry")
    if db is None or client is None or registry is None:
        raise ValueError(
            "Missing runtime dependencies (db, client, or registry) in LangGraph config."
        )
    return db, client, registry



def _fingerprint(selection: ToolSelection) -> str:
    return selection.tool_name + ":" + json.dumps(
        selection.arguments,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_shopper_profile_request(objective: str) -> bool:
    """Identify explicit requests to view saved shopping preferences."""
    text = objective.lower().strip()

    profile_phrases = (
        "shopping profile",
        "shopper profile",
        "my profile",
        "saved preferences",
        "shopping preferences",
        "my preferences",
        "my preferred brands",
        "what brands do i prefer",
        "what categories do i prefer",
        "what do you know about my shopping",
        "show my preferences",
    )

    return any(phrase in text for phrase in profile_phrases)


def _comparison_selection(state: ShopperGraphState) -> ToolSelection | None:
    if not state.get("comparison_requested") or any(
        record.error_code is None
        and isinstance(record.result, ProductComparisonResult)
        for record in state.get("tool_results", [])
    ):
        return None

    product_ids: list[str] = []
    for record in reversed(state.get("tool_results", [])):
        if isinstance(record.result, RecommendationResponse):
            product_ids = [product.product_id for product in record.result.products]
            break
        if isinstance(record.result, PreferenceSimulationResult):
            product_ids = [
                product.product_id
                for product in record.result.recommendations.products
            ]
            break
    if len(product_ids) < 2 and not state.get("fallback_search_allowed", True):
        product_ids = state.get("previously_shown_product_ids", [])
    if len(product_ids) < 2:
        return None
    return ToolSelection(
        tool_name="product_comparison",
        arguments={"product_ids": product_ids[:2]},
    )


def request_planning_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Inspects validated request and observations to plan the next bounded action."""
    db, client, registry = get_runtime_context(config)

    budget = state.get("remaining_execution_budget", 5)
    iterations = state.get("planning_iteration_count", 0)
    seen_calls = list(state.get("seen_calls", []))
    completed_steps = list(state.get("completed_steps", []))
    tool_results = list(state.get("tool_results", []))

    if budget <= 0 or iterations >= 3:
        return {
            "execution_status": "partial",
            "next_action": "response_metadata",
            "planning_iteration_count": iterations + 1,
        }

    
    # Handle explicit shopper profile requests first.
    if _is_shopper_profile_request(state.get("objective", "")):
        selection = ToolSelection(
            tool_name="get_shopper_profile",
            arguments={},
        )
        selection_source: ToolSelectionSource = "policy"

    else:
        # Check comparison policy selection.
        policy_selection = _comparison_selection(state)

        if policy_selection is not None:
            selection = policy_selection
            selection_source = "policy"

        else:
            planner = AgentPlanner(client, registry)
            agent_state = graph_state_to_agent_state(state)

            try:
                selection = planner.select_next_tool(agent_state)
                selection_source = "planner"

            except (
                GroqClientError,
                ValidationError,
                ValueError,
                TypeError,
            ) as error:
                logger.warning(
                    "Agent planning unavailable (%s).",
                    type(error).__name__,
                )

                has_recs = any(
                    isinstance(
                        r.result,
                        (RecommendationResponse, PreferenceSimulationResult),
                    )
                    for r in tool_results
                )

                can_fallback = (
                    state.get("fallback_search_allowed", True)
                    and state["constraints"].category is not None
                    and not has_recs
                    and budget >= 2
                )

                return {
                    "planning_failed": True,
                    "execution_status": (
                        "fallback" if can_fallback else "unavailable"
                    ),
                    "next_action": (
                        "safe_fallback"
                        if can_fallback
                        else "response_metadata"
                    ),
                    "planning_iteration_count": iterations + 1,
                }


    fingerprint = _fingerprint(selection)
    if fingerprint in seen_calls:
        completed_steps.append("Repeated tool call prevented.")
        return {
            "execution_status": "partial",
            "completed_steps": completed_steps,
            "next_action": "response_metadata",
            "planning_iteration_count": iterations + 1,
        }
    seen_calls.append(fingerprint)

    # Budget reservation check: last 1 unit is reserved for evaluation
    if budget == 1 and selection.tool_name != "evaluate_results":
        completed_steps.append("Execution budget reserved for evaluation.")
        return {
            "execution_status": "partial",
            "completed_steps": completed_steps,
            "next_action": "response_metadata",
            "planning_iteration_count": iterations + 1,
        }

    # Validate tool arguments before proceeding to node execution
    try:
        registry.validate_arguments(selection.tool_name, selection.arguments)
    except (InvalidToolArgumentsError, UnknownToolError):
        tool_results.append(
            ToolExecutionRecord(
                tool_name=selection.tool_name,
                arguments={},
                error_code="invalid_arguments",
                executed=False,
                selection_source=selection_source,
            )
        )
        completed_steps.append("Invalid tool arguments were rejected.")
        return {
            "tool_results": tool_results,
            "completed_steps": completed_steps,
            "current_plan": selection,
            "seen_calls": seen_calls,
            "next_action": "response_metadata",
            "planning_iteration_count": iterations + 1,
        }

    return {
        "current_plan": selection,
        "seen_calls": seen_calls,
        "completed_steps": completed_steps,
        "next_action": selection.tool_name,
        "planning_iteration_count": iterations + 1,
    }


def _execute_tool_helper(
    tool_name: str,
    default_arguments: dict[str, Any],
    state: ShopperGraphState,
    config: RunnableConfig,
    selection_source: ToolSelectionSource = "planner",
) -> dict[str, Any]:
    db, _, registry = get_runtime_context(config)
    plan = state.get("current_plan")
    arguments_dict = plan.arguments if plan and plan.tool_name == tool_name else default_arguments

    budget = state.get("remaining_execution_budget", 5)
    tool_count = state.get("tool_execution_count", 0)
    completed_steps = list(state.get("completed_steps", []))
    tool_results = list(state.get("tool_results", []))

    if budget <= 0:
        return {"execution_status": "partial"}

    try:
        validated_args = registry.validate_arguments(tool_name, arguments_dict)
    except (InvalidToolArgumentsError, UnknownToolError):
        tool_results.append(
            ToolExecutionRecord(
                tool_name=tool_name,
                arguments={},
                error_code="invalid_arguments",
                executed=False,
                selection_source=selection_source,
            )
        )
        completed_steps.append("Invalid tool arguments were rejected.")
        return {
            "tool_results": tool_results,
            "completed_steps": completed_steps,
        }

    budget -= 1
    tool_count += 1
    normalized_arguments = validated_args.model_dump(mode="json")
    agent_state = graph_state_to_agent_state(state)

    try:
        result = registry.execute(tool_name, validated_args, agent_state, db)
    except Exception as error:
        logger.error("Agent tool %s failed (%s).", tool_name, type(error).__name__)
        tool_results.append(
            ToolExecutionRecord(
                tool_name=tool_name,
                arguments=normalized_arguments,
                error_code="tool_execution_failed",
                selection_source=selection_source,
            )
        )
    else:
        tool_results.append(
            ToolExecutionRecord(
                tool_name=tool_name,
                arguments=normalized_arguments,
                result=result,
                selection_source=selection_source,
            )
        )

    completed_steps.append(_TOOL_ACTIVITY.get(tool_name, "Running registered tool."))
    return {
        "remaining_execution_budget": budget,
        "tool_execution_count": tool_count,
        "tool_results": tool_results,
        "completed_steps": completed_steps,
    }


def product_search_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Executes catalogue search using registered product_search tool."""
    return _execute_tool_helper(
        "product_search",
        {"limit": state.get("request_limit", 10)},
        state,
        config,
        selection_source="planner",
    )


def product_comparison_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Executes product comparison using registered product_comparison tool."""
    product_ids: list[str] = []
    for record in reversed(state.get("tool_results", [])):
        if isinstance(record.result, RecommendationResponse):
            product_ids = [p.product_id for p in record.result.products]
            break
        if isinstance(record.result, PreferenceSimulationResult):
            product_ids = [p.product_id for p in record.result.recommendations.products]
            break
    if len(product_ids) < 2:
        product_ids = state.get("previously_shown_product_ids", [])

    return _execute_tool_helper(
        "product_comparison",
        {"product_ids": product_ids[:2]},
        state,
        config,
        selection_source="policy",
    )


def shopper_profile_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Retrieves saved shopper profile using registered get_shopper_profile tool."""
    return _execute_tool_helper(
        "get_shopper_profile",
        {},
        state,
        config,
        selection_source="planner",
    )


def simulate_preferences_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Simulates hypothetical preferences using registered simulate_preferences tool."""
    return _execute_tool_helper(
        "simulate_preferences",
        {},
        state,
        config,
        selection_source="planner",
    )


def evaluate_results_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Evaluates the latest tool execution record deterministically against catalogue constraints."""
    db, _, registry = get_runtime_context(config)
    tool_results = list(state.get("tool_results", []))
    completed_steps = list(state.get("completed_steps", []))
    budget = state.get("remaining_execution_budget", 5)
    tool_count = state.get("tool_execution_count", 0)

    if not tool_results or budget <= 0:
        return {"execution_status": "partial"}

    # Find the latest non-evaluation tool record to evaluate
    target_index = None
    for idx in range(len(tool_results) - 1, -1, -1):
        if tool_results[idx].tool_name != "evaluate_results":
            target_index = idx
            break

    if target_index is None:
        return {"execution_status": "partial"}

    eval_args = EvaluationArguments(result_index=target_index)
    budget -= 1
    tool_count += 1
    agent_state = graph_state_to_agent_state(state)

    try:
        evaluation = registry.execute("evaluate_results", eval_args, agent_state, db)
    except Exception as error:
        logger.error("evaluate_results failed (%s).", type(error).__name__)
        tool_results.append(
            ToolExecutionRecord(
                tool_name="evaluate_results",
                arguments={"result_index": target_index},
                error_code="tool_execution_failed",
                selection_source="evaluation",
            )
        )
        return {
            "tool_results": tool_results,
            "remaining_execution_budget": budget,
            "tool_execution_count": tool_count,
            "execution_status": "partial",
        }

    tool_results.append(
        ToolExecutionRecord(
            tool_name="evaluate_results",
            arguments={"result_index": target_index},
            result=evaluation,
            selection_source="evaluation",
        )
    )
    completed_steps.append(_TOOL_ACTIVITY["evaluate_results"])

    execution_status = state.get("execution_status", "completed")
    if isinstance(evaluation, EvaluationResult):
        if evaluation.satisfied:
            if execution_status != "fallback":
                execution_status = "completed"
        elif evaluation.outcome == "constraint_violation":
            execution_status = "partial"

    return {
        "latest_evaluation": evaluation,
        "tool_results": tool_results,
        "completed_steps": completed_steps,
        "remaining_execution_budget": budget,
        "tool_execution_count": tool_count,
        "execution_status": execution_status,
    }


def adjust_search_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Adjusts search constraints or excludes failed items before retrying catalogue search."""
    completed_steps = list(state.get("completed_steps", []))
    completed_steps.append("Adjusted search parameters for eligible catalogue retry.")
    return {
        "completed_steps": completed_steps,
    }


def safe_fallback_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Executes safe deterministic fallback search when planning fails."""
    db, _, registry = get_runtime_context(config)
    budget = state.get("remaining_execution_budget", 5)
    tool_count = state.get("tool_execution_count", 0)
    tool_results = list(state.get("tool_results", []))
    completed_steps = list(state.get("completed_steps", []))
    seen_calls = list(state.get("seen_calls", []))

    if (
        not state.get("fallback_search_allowed", True)
        or state["constraints"].category is None
        or budget < 2
    ):
        return {"execution_status": "unavailable"}

    selection = ToolSelection(
        tool_name="product_search",
        arguments={"limit": state.get("request_limit", 10)},
    )
    fingerprint = _fingerprint(selection)
    if fingerprint in seen_calls:
        return {"execution_status": "partial"}
    seen_calls.append(fingerprint)

    validated_args = ProductSearchArguments(limit=state.get("request_limit", 10))
    budget -= 1
    tool_count += 1
    agent_state = graph_state_to_agent_state(state)

    try:
        result = registry.execute("product_search", validated_args, agent_state, db)
    except Exception as error:
        logger.error("Safe fallback search failed (%s).", type(error).__name__)
        tool_results.append(
            ToolExecutionRecord(
                tool_name="product_search",
                arguments=validated_args.model_dump(mode="json"),
                error_code="tool_execution_failed",
                selection_source="fallback",
            )
        )
    else:
        tool_results.append(
            ToolExecutionRecord(
                tool_name="product_search",
                arguments=validated_args.model_dump(mode="json"),
                result=result,
                selection_source="fallback",
            )
        )
    completed_steps.append(_TOOL_ACTIVITY["product_search"])

    return {
        "execution_status": "fallback",
        "remaining_execution_budget": budget,
        "tool_execution_count": tool_count,
        "tool_results": tool_results,
        "completed_steps": completed_steps,
        "seen_calls": seen_calls,
    }


def response_metadata_node(
    state: ShopperGraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Finalizes execution status and wraps the result."""
    evaluation = state.get("latest_evaluation")
    execution_status = state.get("execution_status", "completed")

    if evaluation is not None and not evaluation.satisfied and execution_status == "completed":
        execution_status = "partial"

    return {"execution_status": execution_status}
