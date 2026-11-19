"""AI shopping assistant API tests using mocked Groq HTTP transport."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.assistant as assistant_api
import app.services.assistant_response as assistant_response_service
from app.core.config import Settings
from app.models.product import Product
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse, AssistantIntent
from app.services.assistant_response import DEMO_CATALOGUE_DISCLAIMER
from app.services.groq_client import GroqClient


def _create_user(client: TestClient, payload: dict) -> int:
    response = client.post("/api/profiles", json=payload)
    assert response.status_code == 201
    return response.json()["id"]


def _install_mock_groq(
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_response: bool = False,
) -> list[dict]:
    calls: list[dict] = []
    intent = {
        "category": "Headphones",
        "max_budget_override": 6000,
        "min_budget_override": None,
        "brand": None,
        "min_rating": None,
        "min_battery_life_hours": None,
        "require_battery_life_data": None,
        "availability_requirement": None,
        "comfort_priority": 85,
        "connectivity_requirement": "wireless",
        "portability_priority": None,
        "unsupported_requirements": [],
        "product_type_requirement": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        messages = body["messages"]
        if invalid_response:
            content = "{invalid JSON"
        elif "Extract shopping intent only." in messages[0]["content"]:
            content = json.dumps(intent)
        elif "Select exactly one next action from the registered tools." in messages[0]["content"]:
            planning_data = json.loads(messages[1]["content"])
            content = json.dumps(
                {
                    "tool_name": "product_search",
                    "arguments": {
                        "limit": min(5, planning_data["requested_limit"]),
                    },
                }
            )
        else:
            request_data = json.loads(messages[1]["content"])
            content = json.dumps(
                {
                    "summary": "These results reflect the requested category and priorities.",
                    "product_explanations": [
                        {
                            "product_id": product["product_id"],
                            "explanation": (
                                "Its listed reasons align with the stated preferences "
                                "and requested category."
                            ),
                        }
                        for product in request_data["recommendations"]
                    ],
                }
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    client = GroqClient(
        settings=Settings(
            _env_file=None,
            groq_api_key="test-only-key",
            groq_model="test-model",
        ),
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(assistant_response_service, "GroqClient", lambda: client)
    return calls


def _disable_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GroqClient(settings=Settings(_env_file=None, groq_api_key=None))
    monkeypatch.setattr(assistant_response_service, "GroqClient", lambda: client)


def _add_headphones(
    db_session: Session,
    *,
    product_id: str,
    price: float,
    connectivity: str,
    brand: str = "TestAudio",
    rating: float = 4.0,
    attribute_score: float | None = 70,
    battery_life_hours: float | None = 20,
) -> None:
    db_session.add(
        Product(
            product_id=product_id,
            name=f"Test headphones {product_id}",
            brand=brand,
            category="Headphones",
            price=price,
            rating=rating,
            description="Illustrative test product.",
            attributes={
                "connectivity": connectivity,
                "battery_life_hours": battery_life_hours,
                "comfort_score": attribute_score,
                "portability": attribute_score,
                "build_quality": attribute_score,
            },
            stock_quantity=10,
            is_active=True,
            image_url=None,
        )
    )
    db_session.commit()


def test_assistant_chat_returns_structured_grounded_response(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    calls = _install_mock_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless comfortable earbuds under ₹6,000.",
            "limit": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == user_id
    assert body["answer"].startswith("Its listed reasons align")
    assert DEMO_CATALOGUE_DISCLAIMER in body["answer"]
    assert body["intent"] == {
        "category": "Headphones",
        "max_budget_override": 6000,
        "min_budget_override": None,
        "brand": None,
        "min_rating": None,
        "min_battery_life_hours": None,
        "require_battery_life_data": None,
        "availability_requirement": None,
        "comfort_priority": 85,
        "brand_loyalty": None,
        "battery_life_priority": None,
        "portability_priority": None,
        "connectivity_requirement": "wireless",
        "product_type_requirement": "earbuds",
        "unsupported_requirements": [],
    }
    assert body["recommendations"]
    assert [product["product_id"] for product in body["recommendations"]] == [
        "SS-HP-002"
    ]
    assert len(body["recommendations"]) <= 2
    assert all(
        {
            "product_id",
            "name",
            "brand",
            "price",
            "rating",
            "score",
            "reasons",
        }
        <= product.keys()
        for product in body["recommendations"]
    )
    assert body["fallback_used"] is False
    assert body["kind"] == "shopping"
    assert body["has_more"] is False
    assert not any("Unverified request details" in note for note in body["notes"])
    assert len(calls) == 3


def test_initial_request_preserves_hard_budget_connectivity_and_battery_constraints(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _disable_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless earbuds with long battery life under ₹5,000.",
            "limit": 10,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["max_budget_override"] == 5000
    assert body["intent"]["connectivity_requirement"] == "wireless"
    assert body["intent"]["product_type_requirement"] == "earbuds"
    assert body["intent"]["battery_life_priority"] == 80
    assert body["recommendations"]
    assert all(product["price"] <= 5000 for product in body["recommendations"])
    assert [product["product_id"] for product in body["recommendations"]] == [
        "SS-HP-002"
    ]


def test_category_only_request_asks_a_useful_clarifying_question(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _disable_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={"user_id": user_id, "message": "I'm looking for headphones."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "clarification"
    assert body["recommendations"] == []
    assert "budget" in body["answer"].lower()
    assert "battery life" in body["answer"].lower()


def test_travel_request_and_battery_follow_up_use_catalogue_facts(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-TRAVEL-TEST",
        price=3200,
        connectivity="wireless",
        battery_life_hours=24,
        attribute_score=92,
    )
    _disable_groq(monkeypatch)

    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones for travel under ₹5,000.",
            "limit": 5,
        },
    ).json()

    assert initial["intent"]["portability_priority"] == 80
    assert initial["intent"]["connectivity_requirement"] == "wireless"
    assert all(product["price"] <= 5000 for product in initial["recommendations"])

    follow_up = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "What about battery?",
            "conversation_context": initial["intent"],
            "previous_product_ids": [
                product["product_id"] for product in initial["recommendations"]
            ],
        },
    ).json()

    assert follow_up["kind"] == "answer"
    assert follow_up["recommendations"] == []
    assert "24 hours" in follow_up["answer"]


def test_travel_follow_up_compares_previous_products_without_searching(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-TRAVEL-CONTEXT-A",
        price=3200,
        connectivity="wireless",
        attribute_score=92,
    )
    _add_headphones(
        db_session,
        product_id="SS-HP-TRAVEL-CONTEXT-B",
        price=4100,
        connectivity="wireless",
        attribute_score=68,
        battery_life_hours=None,
    )
    _disable_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Which of those options would you recommend for travelling?",
            "conversation_context": {
                "category": "Headphones",
                "max_budget_override": 5000,
                "connectivity_requirement": "wireless",
            },
            "previous_product_ids": [
                "SS-HP-TRAVEL-CONTEXT-A",
                "SS-HP-TRAVEL-CONTEXT-B",
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "answer"
    assert "best-supported travel pick" in body["answer"]
    assert "Test headphones SS-HP-TRAVEL-CONTEXT-A" in body["answer"]
    assert body["agent_metadata"]["tools_executed"] == [
        "product_comparison",
        "evaluate_results",
    ]
    assert "product_search" not in body["agent_metadata"]["tools_executed"]


@pytest.mark.parametrize(
    ("message", "expected_product_id"),
    [
        ("Is the first one worth buying?", "SS-HP-REFERENCE-FIRST"),
        ("Is the cheaper one worth buying?", "SS-HP-REFERENCE-CHEAPER"),
    ],
)
def test_contextual_product_references_select_previous_products(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected_product_id: str,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-REFERENCE-FIRST",
        price=4200,
        connectivity="wireless",
    )
    _add_headphones(
        db_session,
        product_id="SS-HP-REFERENCE-CHEAPER",
        price=3200,
        connectivity="wireless",
    )
    _disable_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": message,
            "conversation_context": {
                "category": "Headphones",
                "connectivity_requirement": "wireless",
            },
            "previous_product_ids": [
                "SS-HP-REFERENCE-FIRST",
                "SS-HP-REFERENCE-CHEAPER",
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "answer"
    assert f"Test headphones {expected_product_id}" in body["answer"]
    other_product_id = (
        "SS-HP-REFERENCE-CHEAPER"
        if expected_product_id == "SS-HP-REFERENCE-FIRST"
        else "SS-HP-REFERENCE-FIRST"
    )
    assert f"Test headphones {other_product_id}" not in body["answer"]


def test_comparison_and_why_follow_ups_use_previous_catalogue_products(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-COMPARE",
        price=3800,
        connectivity="wireless",
    )
    _disable_groq(monkeypatch)

    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones and earbuds under ₹5,000.",
            "limit": 5,
        },
    ).json()
    previous_product_ids = [
        product["product_id"] for product in initial["recommendations"]
    ]
    assert len(previous_product_ids) >= 2

    comparison = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Compare these products.",
            "conversation_context": initial["intent"],
            "previous_product_ids": previous_product_ids,
        },
    ).json()
    why = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Why was PulseBuds Compact recommended?",
            "conversation_context": initial["intent"],
            "previous_product_ids": previous_product_ids,
        },
    ).json()

    assert comparison["kind"] == "answer"
    assert comparison["recommendations"] == []
    assert all(
        product["name"] in comparison["answer"]
        for product in initial["recommendations"]
    )
    assert all(
        f"₹{product['price']:,.0f}" in comparison["answer"]
        for product in initial["recommendations"]
    )
    assert why["kind"] == "answer"
    assert why["recommendations"] == []
    assert "PulseBuds Compact" in why["answer"]


def test_natural_product_questions_compare_only_listed_catalogue_facts(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-QUESTION-LISTED",
        price=3800,
        connectivity="wireless",
        attribute_score=90,
        battery_life_hours=30,
    )
    _add_headphones(
        db_session,
        product_id="SS-HP-QUESTION-MISSING",
        price=3200,
        connectivity="wireless",
        attribute_score=None,
        battery_life_hours=None,
    )
    calls = _install_mock_groq(monkeypatch)
    context = AssistantIntent(
        category="Headphones",
        connectivity_requirement="wireless",
    )
    previous_product_ids = [
        "SS-HP-QUESTION-LISTED",
        "SS-HP-QUESTION-MISSING",
    ]

    battery = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Which one has better battery life?",
            "conversation_context": context.model_dump(),
            "previous_product_ids": previous_product_ids,
        },
    ).json()
    travel = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Which one is better for travel?",
            "conversation_context": context.model_dump(),
            "previous_product_ids": previous_product_ids,
        },
    ).json()
    comparison = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Compare these two.",
            "conversation_context": context.model_dump(),
            "previous_product_ids": previous_product_ids,
        },
    ).json()
    disadvantages = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "What are the disadvantages?",
            "conversation_context": context.model_dump(),
            "previous_product_ids": previous_product_ids,
        },
    ).json()
    worth = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Is this worth buying?",
            "conversation_context": context.model_dump(),
            "previous_product_ids": previous_product_ids,
        },
    ).json()

    assert battery["kind"] == "answer"
    assert (
        "Test headphones SS-HP-QUESTION-LISTED is the only option with listed "
        "battery data"
    ) in battery["answer"]
    assert "I can't compare it with the other unlisted durations" in battery["answer"]
    assert (
        "does not list battery life for Test headphones SS-HP-QUESTION-MISSING"
        in battery["answer"]
    )
    assert travel["kind"] == "answer"
    assert "don't establish real-world travel suitability" in travel["answer"]
    assert "travel portability not listed" in travel["answer"]
    assert comparison["kind"] == "answer"
    assert "battery life not listed" in comparison["answer"]
    assert "don't establish an overall winner" in comparison["answer"]
    assert disadvantages["kind"] == "answer"
    assert "doesn't record verified drawbacks" in disadvantages["answer"]
    assert worth["kind"] == "answer"
    assert "can't determine whether" in worth["answer"]
    assert all(
        response["recommendations"] == []
        for response in (battery, travel, disadvantages, worth)
    )
    assert calls == []


def test_ambiguous_broad_followup_asks_before_relaxing_filters(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    calls = _install_mock_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Show me anything.",
            "conversation_context": AssistantIntent(
                category="Headphones",
                max_budget_override=5000,
                min_battery_life_hours=20,
                require_battery_life_data=True,
                connectivity_requirement="wireless",
            ).model_dump(),
            "previous_product_ids": ["SS-HP-002"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "clarification"
    assert body["recommendations"] == []
    assert "relax the budget" in body["answer"]
    assert "battery-life requirement" in body["answer"]
    assert "wireless requirement" in body["answer"]
    assert "product category" in body["answer"]
    assert "I'll keep your current filters" in body["answer"]
    assert calls == []


def test_brand_followup_retains_category_budget_and_connectivity(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-SONY-FOLLOWUP",
        price=4200,
        connectivity="wireless",
        brand="Sony",
    )
    _disable_groq(monkeypatch)

    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones under ₹5,000.",
        },
    ).json()
    followup = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Anything from Sony?",
            "conversation_context": initial["intent"],
            "previous_product_ids": [
                product["product_id"] for product in initial["recommendations"]
            ],
        },
    ).json()

    assert followup["intent"]["category"] == "Headphones"
    assert followup["intent"]["max_budget_override"] == 5000
    assert followup["intent"]["connectivity_requirement"] == "wireless"
    assert followup["intent"]["brand"] == "Sony"
    assert [product["product_id"] for product in followup["recommendations"]] == [
        "SS-HP-SONY-FOLLOWUP"
    ]


def test_new_product_category_does_not_inherit_unrelated_constraints(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _disable_groq(monkeypatch)

    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones under ₹5,000.",
        },
    ).json()
    changed_category = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Laptops under ₹15,000.",
            "conversation_context": initial["intent"],
            "previous_product_ids": [
                product["product_id"] for product in initial["recommendations"]
            ],
        },
    ).json()

    assert changed_category["intent"]["category"] == "Laptops"
    assert changed_category["intent"]["max_budget_override"] == 15000
    assert changed_category["intent"]["connectivity_requirement"] is None
    assert changed_category["intent"]["product_type_requirement"] is None


def test_show_more_paginates_without_repeating_and_keeps_original_filters(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    for index, price in enumerate((2800, 3200, 3600, 4200, 4600), start=1):
        _add_headphones(
            db_session,
            product_id=f"SS-HP-PAGE-{index:03d}",
            price=price,
            connectivity="wireless",
        )
    _add_headphones(
        db_session,
        product_id="SS-HP-OVER-BUDGET",
        price=5999,
        connectivity="wireless",
    )
    _disable_groq(monkeypatch)

    first = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones and earbuds with long battery life under ₹5,000.",
            "limit": 2,
        },
    ).json()
    first_ids = [product["product_id"] for product in first["recommendations"]]
    assert len(first_ids) == 2
    assert first["has_more"] is True

    second = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Show me more options.",
            "limit": 2,
            "conversation_context": first["intent"],
            "previous_product_ids": first_ids,
        },
    ).json()
    second_ids = [product["product_id"] for product in second["recommendations"]]

    assert second["kind"] == "shopping"
    assert second["intent"] == first["intent"]
    assert second_ids
    assert set(first_ids).isdisjoint(second_ids)
    assert all(product["price"] <= 5000 for product in second["recommendations"])
    assert second["has_more"] is True

    third = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Show me more options.",
            "limit": 2,
            "conversation_context": second["intent"],
            "previous_product_ids": first_ids + second_ids,
        },
    ).json()
    third_ids = [product["product_id"] for product in third["recommendations"]]

    assert third_ids
    assert set(first_ids + second_ids).isdisjoint(third_ids)
    assert third["has_more"] is False
    assert all(product["price"] <= 5000 for product in third["recommendations"])


def test_explicitly_switching_to_wired_keeps_budget_and_battery_priority(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-WIRED-UNDER-BUDGET",
        price=3200,
        connectivity="wired",
    )
    _disable_groq(monkeypatch)

    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones and earbuds with long battery life under ₹5,000.",
        },
    ).json()
    changed = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Actually, show me wired headphones instead.",
            "conversation_context": initial["intent"],
            "previous_product_ids": [
                product["product_id"] for product in initial["recommendations"]
            ],
        },
    ).json()

    assert changed["intent"]["category"] == "Headphones"
    assert changed["intent"]["connectivity_requirement"] == "wired"
    assert changed["intent"]["max_budget_override"] == 5000
    assert changed["intent"]["battery_life_priority"] == 80
    assert [product["product_id"] for product in changed["recommendations"]] == [
        "SS-HP-WIRED-UNDER-BUDGET"
    ]
    assert all(product["price"] <= 5000 for product in changed["recommendations"])


def test_new_search_resets_stale_filters_from_previous_search(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)

    _add_headphones(
        db_session,
        product_id="SS-HP-FRESH-SEARCH",
        price=4000,
        connectivity="wireless",
        brand="TestAudio",
        rating=4.0,
        battery_life_hours=10,
    )
    _disable_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Find wireless headphones under ₹5,000.",
            "conversation_context": {
                "category": "Headphones",
                "brand": "Sony",
                "max_budget_override": 6000,
                "min_rating": 4.5,
                "min_battery_life_hours": 20,
                "require_battery_life_data": True,
                "availability_requirement": "available",
                "connectivity_requirement": "wireless",
            },
        },
    )

    assert response.status_code == 200
    intent = response.json()["intent"]

    assert intent["category"] == "Headphones"
    assert intent["max_budget_override"] == 5000
    assert intent["brand"] is None
    assert intent["min_rating"] is None
    assert intent["min_battery_life_hours"] is None
    assert intent["require_battery_life_data"] is None
    assert intent["availability_requirement"] is None
    assert intent["connectivity_requirement"] == "wireless"

def test_explicit_budget_follow_up_narrows_budget_and_retains_connectivity(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _disable_groq(monkeypatch)
    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones and earbuds with long battery life under ₹5,000.",
        },
    ).json()

    narrowed = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Lower the budget to ₹3,000.",
            "conversation_context": initial["intent"],
            "previous_product_ids": [
                product["product_id"] for product in initial["recommendations"]
            ],
        },
    ).json()

    assert narrowed["intent"]["max_budget_override"] == 3000
    assert narrowed["intent"]["connectivity_requirement"] == "wireless"
    assert narrowed["intent"]["battery_life_priority"] == 80
    assert all(product["price"] <= 3000 for product in narrowed["recommendations"])


def test_cheaper_follow_up_uses_previous_prices_and_retains_other_constraints(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _add_headphones(
        db_session,
        product_id="SS-HP-CHEAP-LOW-SCORE",
        price=1500,
        connectivity="wireless",
        rating=1,
        attribute_score=5,
        battery_life_hours=1,
    )
    _disable_groq(monkeypatch)
    initial = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones and earbuds with long battery life under ₹5,000.",
            "limit": 1,
        },
    ).json()
    previous_ids = [
        product["product_id"] for product in initial["recommendations"]
    ]
    previous_max_price = max(
        product["price"] for product in initial["recommendations"]
    )

    cheaper = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Show me cheaper ones.",
            "conversation_context": initial["intent"],
            "previous_product_ids": previous_ids,
        },
    ).json()

    assert cheaper["intent"]["max_budget_override"] < previous_max_price
    assert cheaper["intent"]["connectivity_requirement"] == "wireless"
    assert cheaper["intent"]["battery_life_priority"] == 80
    assert cheaper["recommendations"]
    assert set(previous_ids).isdisjoint(
        product["product_id"] for product in cheaper["recommendations"]
    )
    assert all(
        product["price"] < previous_max_price
        for product in cheaper["recommendations"]
    )


@pytest.mark.parametrize(
    ("message", "expected_start"),
    [
        ("Thank you!", "You're welcome!"),
        ("Hello.", "Hello!"),
        ("Good morning.", "Good morning!"),
        ("Okay", "Glad that helped."),
        ("Got it.", "Glad that helped."),
        ("That's helpful!", "You're welcome!"),
    ],
)
def test_acknowledgements_and_greetings_do_not_ask_for_category_or_recommend(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected_start: str,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    calls = _install_mock_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": message,
            "conversation_context": AssistantIntent(
                category="Headphones",
                max_budget_override=5000,
                connectivity_requirement="wireless",
            ).model_dump(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "acknowledgement"
    assert body["answer"].startswith(expected_start)
    assert body["recommendations"] == []
    assert "clarify" not in body["answer"].lower()
    assert "couldn't find" not in body["answer"].lower()
    assert calls == []


def test_no_eligible_products_are_reported_without_weakening_constraints(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    _disable_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={
            "user_id": user_id,
            "message": "Wireless headphones under ₹1,200.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["intent"]["max_budget_override"] == 1200
    assert body["intent"]["connectivity_requirement"] == "wireless"
    assert "in the current catalogue" in body["answer"]
    assert "wireless connectivity" in body["answer"]
    assert "kept your filters intact" in body["answer"]
    assert "raising the budget" in body["answer"]
    assert body["has_more"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user_id": 1},
        {"message": "Headphones"},
        {"user_id": 0, "message": "Headphones"},
        {"user_id": 1, "message": "   "},
        {"user_id": 1, "message": "Headphones", "limit": 26},
    ],
)
def test_assistant_chat_rejects_missing_or_invalid_request_fields(
    client: TestClient,
    payload: dict,
) -> None:
    response = client.post("/api/assistant/chat", json=payload)

    assert response.status_code == 422


def test_assistant_chat_returns_not_found_for_nonexistent_user(
    seeded_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_mock_groq(monkeypatch)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={"user_id": 9999, "message": "Headphones."},
    )

    assert response.status_code == 404
    assert len(calls) == 0


def test_assistant_chat_uses_deterministic_fallback_for_invalid_groq_response(
    seeded_client: TestClient,
    sample_profile_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = _create_user(seeded_client, sample_profile_payload)
    calls = _install_mock_groq(monkeypatch, invalid_response=True)

    response = seeded_client.post(
        "/api/assistant/chat",
        json={"user_id": user_id, "message": "Headphones under ₹6,000."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["recommendations"]
    assert "stay within your current filters" in body["answer"]
    assert DEMO_CATALOGUE_DISCLAIMER in body["answer"]
    assert any("deterministic explanation" in note for note in body["notes"])
    assert len(calls) == 3


def test_assistant_chat_delegates_to_existing_response_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Session, AssistantChatRequest]] = []
    expected = AssistantChatResponse(
        user_id=42,
        answer="Service-generated response.",
        intent=AssistantIntent(category="Headphones"),
        recommendations=[],
        fallback_used=True,
        notes=["service note"],
    )

    def service_spy(
        db: Session,
        payload: AssistantChatRequest,
    ) -> AssistantChatResponse:
        calls.append((db, payload))
        return expected

    monkeypatch.setattr(assistant_api, "generate_assistant_response", service_spy)
    response = client.post(
        "/api/assistant/chat",
        json={"user_id": 42, "message": "Headphones."},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == expected.answer
    assert response.json()["notes"] == expected.notes
    assert len(calls) == 1
    assert isinstance(calls[0][0], Session)
    assert calls[0][1] == AssistantChatRequest(user_id=42, message="Headphones.")
