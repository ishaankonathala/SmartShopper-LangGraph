"""Conditional edge routing for the SmartShopper LangGraph workflow."""

from __future__ import annotations

import json
from typing import Literal
from app.agent.schemas import EvaluationResult, ProductComparisonResult, ToolSelection
from app.agent.state import ShopperGraphState
from app.schemas.recommendation import RecommendationResponse
from app.agent.schemas import PreferenceSimulationResult


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
            product_ids = [p.product_id for p in record.result.products]
            break
        if isinstance(record.result, PreferenceSimulationResult):
            product_ids = [
                p.product_id
                for p in record.result.recommendations.products
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


def route_after_planning(state: ShopperGraphState) -> str:
    """Route from the planning node to the planned tool node or fallback/completion."""
    next_action = state.get("next_action")
    if next_action in {
        "product_search",
        "product_comparison",
        "get_shopper_profile",
        "simulate_preferences",
        "evaluate_results",
        "safe_fallback",
        "response_metadata",
    }:
        return next_action

    plan = state.get("current_plan")
    if plan is not None:
        if plan.tool_name in {
            "product_search",
            "product_comparison",
            "get_shopper_profile",
            "simulate_preferences",
            "evaluate_results",
        }:
            return plan.tool_name

    if state.get("planning_failed") and state.get("fallback_search_allowed"):
        return "safe_fallback"

    return "response_metadata"


def route_after_evaluation(state: ShopperGraphState) -> str:
    """Route from the evaluation node to comparison, retry, planning or final response."""
    evaluation: EvaluationResult | None = state.get("latest_evaluation")
    remaining_budget = state.get("remaining_execution_budget", 0)
    iterations = state.get("planning_iteration_count", 0)

    if evaluation is None:
        return "response_metadata"

    # Check if comparison is requested and eligible
    if state.get("comparison_requested"):
        has_comparison = any(
            record.tool_name == "product_comparison"
            and record.executed
            and record.error_code is None
            and isinstance(record.result, ProductComparisonResult)
            for record in state.get("tool_results", [])
        )
        if not has_comparison:
            comp_selection = _comparison_selection(state)
            if comp_selection is not None and remaining_budget >= 2:
                return "product_comparison"

    if evaluation.satisfied:
        return "response_metadata"

    if evaluation.outcome == "no_matches":
        return "response_metadata"

    if evaluation.outcome == "constraint_violation":
        if remaining_budget >= 2 and iterations < 3:
            return "adjust_search"
        return "response_metadata"

    # If needs comparison or unsatisfied, continue planning if budget allows
    if remaining_budget >= 2 and iterations < 3:
        return "planning"

    return "response_metadata"
