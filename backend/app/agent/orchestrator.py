"""LangGraph-driven bounded shopping assistant agent orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from sqlalchemy.orm import Session

from app.agent.graph import run_shopper_graph
from app.agent.planner import AgentPlanner
from app.agent.schemas import (
    EvaluationResult,
    PreferenceSimulationResult,
    ProductComparisonResult,
    ShopperProfileResult,
)
from app.agent.state import (
    AgentState,
    ShopperGraphState,
    ToolExecutionRecord,
    graph_state_to_agent_state,
)
from app.agent.tool_registry import ToolRegistry
from app.agent.tools import create_tool_registry
from app.schemas.assistant import AgentActivityEvent, AgentExecutionMetadata
from app.schemas.recommendation import RecommendationResponse
from app.services.groq_client import GroqClient

MAX_AGENT_TOOL_EXECUTIONS: Final[int] = 5
MAX_AGENT_SELECTION_ITERATIONS: Final[int] = 3

_TOOL_TITLES: Final[dict[str, str]] = {
    "product_search": "Product Search",
    "product_comparison": "Product Comparison",
    "get_shopper_profile": "Shopper Profile",
    "simulate_preferences": "Preference Simulation",
    "evaluate_results": "Recommendation Evaluation",
}
_TOOL_ACTIVITY: Final[dict[str, str]] = {
    "product_search": "Searching eligible catalogue products.",
    "product_comparison": "Comparing available catalogue fields.",
    "get_shopper_profile": "Checking your saved shopping preferences.",
    "simulate_preferences": "Simulating temporary preference changes.",
    "evaluate_results": "Checking results against your requirements.",
}
logger = logging.getLogger(__name__)


def _intent_summary(state: AgentState) -> str:
    intent = state.constraints
    constraints: list[str] = []
    if intent.category:
        constraints.append(f"category {intent.category}")
    if intent.max_budget_override is not None:
        constraints.append(f"maximum budget ₹{intent.max_budget_override:,.0f}")
    if intent.min_budget_override is not None:
        constraints.append(f"minimum budget ₹{intent.min_budget_override:,.0f}")
    if intent.brand:
        constraints.append(f"brand {intent.brand}")
    if intent.min_rating is not None:
        constraints.append(f"minimum rating {intent.min_rating:g}/5")
    if intent.connectivity_requirement:
        constraints.append(f"{intent.connectivity_requirement} connectivity")
    if intent.product_type_requirement:
        constraints.append(f"product type {intent.product_type_requirement}")
    if intent.min_battery_life_hours is not None:
        constraints.append(
            f"at least {intent.min_battery_life_hours:g} hours battery life"
        )
    if intent.availability_requirement:
        constraints.append("verified availability")
    summary = (
        "Applied constraints: " + "; ".join(constraints) + "."
        if constraints
        else "No structured product constraints were available."
    )
    if state.comparison_requested:
        summary += " A product comparison was requested."
    if state.previously_shown_product_ids:
        summary += (
            f" Considering {len(state.previously_shown_product_ids)} "
            "previously shown product(s)."
        )
    return summary[:400]


def _tool_result_summary(record: ToolExecutionRecord) -> str:
    result = record.result
    if record.error_code is not None or not record.executed:
        return "The tool did not complete successfully; error details were not exposed."
    if isinstance(result, RecommendationResponse):
        products = [
            f"{product.name} (₹{product.price:,.0f}, rating {product.rating:g}/5)"
            for product in result.products[:3]
        ]
        summary = f"Returned {len(result.products)} eligible product(s)."
        if products:
            summary += " Shortlist: " + "; ".join(products) + "."
        if result.has_more:
            summary += " More eligible options are available."
        if any(product.catalogue_source == "amazon" for product in result.products):
            summary += (
                " Amazon prices are historical and illustrative; availability is unverified."
            )
        return summary[:400]
    if isinstance(result, PreferenceSimulationResult):
        return (
            f"Simulated preferences and returned "
            f"{len(result.recommendations.products)} eligible product(s). "
            "Saved profile and Shopper DNA were not changed."
        )
    if isinstance(result, ProductComparisonResult):
        products: list[str] = []
        for product in result.products:
            values: list[str] = []
            for field, label in (
                ("price", "₹"),
                ("rating", "rating "),
                ("battery_life_hours", "battery "),
                ("comfort_score", "comfort "),
                ("portability", "portability "),
            ):
                value = product.fields.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if field == "price":
                    values.append(f"₹{value:,.0f}")
                elif field == "rating":
                    values.append(f"rating {value:g}/5")
                elif field == "battery_life_hours":
                    values.append(f"battery {value:g} h")
                else:
                    values.append(f"{label}{value:g}/100")
            detail = f" ({', '.join(values)})" if values else ""
            products.append(f"{product.name}{detail}")
        missing_fields = sorted(
            {
                field.replace("_", " ")
                for fields in result.missing_fields.values()
                for field in fields
            }
        )
        summary = f"Compared {len(result.products)} catalogue product(s)"
        if products:
            summary += ": " + "; ".join(products)
        if missing_fields:
            summary += ". Unavailable catalogue fields: " + ", ".join(missing_fields)
        return summary[:397] + "..." if len(summary) > 400 else summary + "."
    if isinstance(result, ShopperProfileResult):
        return (
            "Retrieved the saved profile and "
            f"{len(result.inferred_preferences)} Shopper DNA preference dimension(s)."
        )
    if isinstance(result, EvaluationResult):
        if result.outcome == "no_matches":
            summary = "No products met every filter; no strict constraint was relaxed."
            if result.missing_specifications:
                missing = ", ".join(
                    field.replace("_", " ")
                    for field in result.missing_specifications
                )
                summary += f" Missing catalogue data: {missing}."
            return summary
        if result.outcome == "needs_comparison":
            return (
                f"Validated {result.eligible_product_count} eligible product(s); "
                "a requested comparison is still needed."
            )
        if result.outcome == "constraint_violation":
            details = list(result.issues[:2])
            if result.missing_specifications:
                missing = ", ".join(
                    field.replace("_", " ")
                    for field in result.missing_specifications
                )
                details.append(f"Missing catalogue data: {missing}.")
            return (
                "The returned products did not pass all strict constraint checks. "
                + " ".join(details)
            )[:400]
        if result.outcome == "comparison_complete":
            return "The comparison passed duplicate and catalogue checks."
        if result.outcome == "eligible_results":
            return (
                f"{result.eligible_product_count} eligible product(s) passed "
                "the strict catalogue checks."
            )
        if result.outcome == "profile_retrieved":
            return "The profile result was verified for this request."
        if result.outcome == "simulated_results":
            return "The temporary preference simulation passed catalogue checks."
        return f"Evaluation outcome: {result.outcome.replace('_', ' ')}."
    return "The tool completed without a displayable result summary."


def _activity_events(state: AgentState) -> list[AgentActivityEvent]:
    events = [
        AgentActivityEvent(
            sequence=1,
            kind="understanding",
            title="Understanding the request",
            status="completed",
            summary=_intent_summary(state),
            succeeded=True,
        )
    ]
    selected_records = [
        record
        for record in state.tool_results
        if record.selection_source != "evaluation"
    ]
    events.append(
        AgentActivityEvent(
            sequence=2,
            kind="planning",
            title="Planning the task",
            status=(
                "partial"
                if state.planning_failed or not selected_records
                else "completed"
            ),
            summary=(
                "The planner could not complete; bounded fallback handling was used where safe."
                if state.planning_failed
                else (
                    f"Selected {len(selected_records)} bounded action(s) using "
                    "the validated request and prior tool results."
                    if selected_records
                    else "No tool action was selected for this request."
                )
            ),
            succeeded=not state.planning_failed and bool(selected_records),
        )
    )

    for record in state.tool_results:
        title = _TOOL_TITLES.get(record.tool_name, "Agent Tool")
        succeeded = record.executed and record.error_code is None
        if record.selection_source != "evaluation":
            source_summary = {
                "planner": "Selected by the planner from the registered tools.",
                "policy": "Selected to satisfy an explicit comparison request.",
                "fallback": "Selected by the deterministic safe fallback.",
            }.get(record.selection_source, "Selected from the registered tools.")
            invalid_arguments = record.error_code == "invalid_arguments"
            events.append(
                AgentActivityEvent(
                    sequence=len(events) + 1,
                    kind="selection",
                    title=f"Selected: {title}",
                    status="failed" if invalid_arguments else "completed",
                    summary=(
                        "The selection was rejected because its arguments failed validation."
                        if invalid_arguments
                        else source_summary
                    ),
                    tool_name=record.tool_name,
                    succeeded=not invalid_arguments,
                )
            )
        if record.executed:
            is_evaluation = record.tool_name == "evaluate_results"
            purpose = _TOOL_ACTIVITY.get(
                record.tool_name,
                "Running the registered agent tool.",
            )
            result_summary = _tool_result_summary(record)
            summary = f"{purpose} {result_summary}"
            if len(summary) > 400:
                summary = summary[:397] + "..."
            event_status = "completed" if succeeded else "failed"
            if is_evaluation and isinstance(record.result, EvaluationResult):
                if record.result.outcome == "constraint_violation":
                    event_status = "failed"
                    succeeded = False
                elif record.result.outcome in {
                    "needs_comparison",
                    "missing_result",
                    "tool_failed",
                }:
                    event_status = "partial"
                    succeeded = False
            events.append(
                AgentActivityEvent(
                    sequence=len(events) + 1,
                    kind="evaluation" if is_evaluation else "tool_execution",
                    title=title,
                    status=event_status,
                    summary=summary,
                    tool_name=record.tool_name,
                    succeeded=succeeded,
                )
            )

    evaluation = state.latest_evaluation
    if evaluation is not None and evaluation.outcome == "no_matches":
        completion_status = "completed"
        completion_summary = (
            "The search completed; no eligible product matched the current catalogue and strict filters."
        )
        completion_succeeded = True
    elif state.execution_status == "unavailable":
        completion_status = "failed"
        completion_summary = "No safe agent action could be completed."
        completion_succeeded = False
    elif state.execution_status in {"partial", "fallback"}:
        completion_status = "partial"
        completion_summary = (
            "The request completed with a safe fallback or incomplete agent check."
        )
        completion_succeeded = False
    else:
        completion_status = "completed"
        completion_summary = "The catalogue-checked result is ready."
        completion_succeeded = True
    events.append(
        AgentActivityEvent(
            sequence=len(events) + 1,
            kind="completion",
            title="Recommendation ready",
            status=completion_status,
            summary=completion_summary,
            succeeded=completion_succeeded,
        )
    )
    return events


@dataclass(frozen=True)
class AgentRunResult:
    state: AgentState
    recommendation_response: RecommendationResponse | None
    comparison_result: ProductComparisonResult | None
    profile_result: ShopperProfileResult | None
    simulation_result: PreferenceSimulationResult | None
    evaluation_result: EvaluationResult | None

    @property
    def metadata(self) -> AgentExecutionMetadata:
        activities = list(
            dict.fromkeys(
                _TOOL_ACTIVITY[record.tool_name]
                for record in self.state.tool_results
                if record.executed and record.tool_name in _TOOL_ACTIVITY
            )
        )
        summary = " ".join(activities) or "Using the existing safe fallback."
        return AgentExecutionMetadata(
            execution_id=str(self.state.execution_id),
            status=self.state.execution_status,
            plan_summary=summary,
            tools_executed=[
                record.tool_name
                for record in self.state.tool_results
                if record.executed
            ],
            tool_execution_count=self.state.tool_execution_count,
            evaluation_outcome=(
                self.state.latest_evaluation.outcome
                if self.state.latest_evaluation is not None
                else "not_evaluated"
            ),
            activity=activities,
            events=_activity_events(self.state),
        )


class AgentOrchestrator:
    """Executes the shopping agent workflow compiled in LangGraph."""

    def __init__(
        self,
        client: GroqClient,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._client = client
        self._registry = registry or create_tool_registry()
        self._planner = AgentPlanner(client, self._registry)

    def run(self, state: AgentState, db: Session) -> AgentRunResult:
        state.remaining_execution_budget = min(
            state.remaining_execution_budget,
            MAX_AGENT_TOOL_EXECUTIONS,
        )

        final_graph_state = run_shopper_graph(
            state,
            db,
            self._client,
            self._registry,
        )

        # Update input state in-place to preserve caller references
        final_agent_state = graph_state_to_agent_state(final_graph_state)
        state.current_plan = final_agent_state.current_plan
        state.planning_failed = final_agent_state.planning_failed
        state.completed_steps = final_agent_state.completed_steps
        state.tool_results = final_agent_state.tool_results
        state.remaining_execution_budget = final_agent_state.remaining_execution_budget
        state.tool_execution_count = final_agent_state.tool_execution_count
        state.latest_evaluation = final_agent_state.latest_evaluation
        state.execution_status = final_agent_state.execution_status

        return self._result(state)

    @staticmethod
    def _result(state: AgentState) -> AgentRunResult:
        recommendation_response = None
        recommendation_result_index = None
        comparison_result = None
        profile_result = None
        simulation_result = None
        for index, record in enumerate(state.tool_results):
            if record.error_code is not None:
                continue
            if isinstance(record.result, RecommendationResponse):
                recommendation_response = record.result
                recommendation_result_index = index
            elif isinstance(record.result, ProductComparisonResult):
                comparison_result = record.result
            elif isinstance(record.result, ShopperProfileResult):
                profile_result = record.result
            elif isinstance(record.result, PreferenceSimulationResult):
                simulation_result = record.result
                recommendation_response = record.result.recommendations
                recommendation_result_index = index

        def passes_recommendation_checks(result: EvaluationResult) -> bool:
            return (
                result.satisfied
                and result.outcome in {"eligible_results", "simulated_results"}
            ) or (
                state.comparison_requested
                and result.outcome == "needs_comparison"
                and not result.issues
            )

        recommendation_was_validated = False
        latest_evaluation = state.latest_evaluation
        if (
            recommendation_result_index is not None
            and latest_evaluation is not None
            and latest_evaluation.outcome == "no_matches"
            and recommendation_response is not None
            and not recommendation_response.products
        ):
            recommendation_was_validated = True
        elif recommendation_result_index is not None and latest_evaluation is not None:
            evaluations_for_recommendation = [
                record.result
                for record in state.tool_results
                if record.tool_name == "evaluate_results"
                and record.arguments.get("result_index") == recommendation_result_index
                and isinstance(record.result, EvaluationResult)
            ]
            if latest_evaluation.outcome in {"eligible_results", "simulated_results"}:
                recommendation_was_validated = any(
                    result is latest_evaluation
                    and passes_recommendation_checks(result)
                    for result in evaluations_for_recommendation
                )
            elif latest_evaluation.outcome == "comparison_complete":
                recommendation_was_validated = any(
                    passes_recommendation_checks(result)
                    for result in evaluations_for_recommendation
                )

        if not recommendation_was_validated:
            recommendation_response = None
            simulation_result = None

        return AgentRunResult(
            state=state,
            recommendation_response=recommendation_response,
            comparison_result=comparison_result,
            profile_result=profile_result,
            simulation_result=simulation_result,
            evaluation_result=state.latest_evaluation,
        )
