"""Assistant recommendation tests spanning the curated and Amazon catalogues."""

from sqlalchemy.orm import Session

from app.models.amazon_catalogue_product import AmazonCatalogueProduct
from app.models.product import Product
from app.models.user_profile import UserProfile
from app.schemas.assistant import AssistantChatRequest, AssistantIntent
from app.schemas.recommendation import RecommendationRequest
from app.services.assistant_response import generate_assistant_response
from app.services.assistant_intent import extract_assistant_intent
from app.services.groq_client import GroqAPIKeyMissingError
from app.services.recommendation import (
    explicitly_requests_children_headphones,
    generate_recommendations,
)
from app.services.seed import seed_products


class FailingGroqClient:
    def complete_json(self, _messages):
        raise GroqAPIKeyMissingError()


def _create_profile(db_session: Session) -> int:
    profile = UserProfile(
        name="Catalogue test",
        max_budget=10_000,
        min_budget=0,
        preferred_categories=[],
        preferred_brands=[],
        disliked_attributes=[],
    )
    db_session.add(profile)
    db_session.commit()
    return profile.id


def _add_amazon_product(
    db_session: Session,
    *,
    asin: str,
    title: str,
    category: str = "Headphones & Earbuds",
    store: str | None = "Sony",
    price_inr: int = 2500,
    rating: float = 4.6,
    rating_count: int = 100,
    features: list[str] | None = None,
) -> AmazonCatalogueProduct:
    product = AmazonCatalogueProduct(
        asin=asin,
        title=title,
        store=store,
        category=category,
        price_inr=price_inr,
        rating=rating,
        rating_count=rating_count,
        image_url="https://m.media-amazon.com/images/I/41image.jpg",
        description=["Imported historical product description."],
        features=features or [],
        source_metadata={
            "source": "Amazon Reviews 2023 Electronics metadata",
            "original_price_usd": "29.41",
            "usd_to_inr_rate": "85",
        },
    )
    db_session.add(product)
    db_session.commit()
    return product


def _add_curated_battery_product(
    db_session: Session,
    *,
    sku: str,
    battery_life_hours: float | str | None,
) -> Product:
    attributes = {
        "connectivity": "wireless",
        "product_type": "headphones",
    }
    if battery_life_hours is not None:
        attributes["battery_life_hours"] = battery_life_hours
    product = Product(
        product_id=sku,
        name=f"TestAudio {sku} Wireless Headphones",
        brand="TestAudio",
        category="Headphones",
        price=2500,
        rating=4.5,
        description="",
        attributes=attributes,
        stock_quantity=5,
        is_active=True,
    )
    db_session.add(product)
    db_session.commit()
    return product


def test_curated_catalogue_still_produces_recommendations(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)

    response = generate_recommendations(
        db_session,
        RecommendationRequest(user_id=user_id, category="Headphones", limit=10),
    )

    assert response.products
    assert all(product.catalogue_source == "curated" for product in response.products)
    assert all(not product.price_is_historical for product in response.products)


def test_headphone_title_signals_only_prioritize_children_on_explicit_request(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    child_headphones = _add_amazon_product(
        db_session,
        asin="A000000001",
        title="iRAG J01 Kids Headphones Wired",
    )
    general_headphones = _add_amazon_product(
        db_session,
        asin="Z000000001",
        title="Sony Wireless Over-Ear Headphones",
    )
    request = RecommendationRequest(
        user_id=user_id,
        category="Headphones",
        limit=2,
        max_budget_override=5000,
        include_unverified_availability=True,
    )

    general_response = generate_recommendations(
        db_session,
        request,
        required_product_type="headphones",
        product_snapshot=[child_headphones, general_headphones],
    )
    kids_request = "Find kids' headphones under ₹5,000."
    children_response = generate_recommendations(
        db_session,
        request,
        required_product_type="headphones",
        prioritize_children_headphones=explicitly_requests_children_headphones(
            kids_request
        ),
        product_snapshot=[child_headphones, general_headphones],
    )

    assert [product.product_id for product in general_response.products] == [
        "amazon:Z000000001",
        "amazon:A000000001",
    ]
    assert [product.product_id for product in children_response.products] == [
        "amazon:A000000001",
        "amazon:Z000000001",
    ]


def test_deterministic_fallback_extracts_imported_category_and_hard_filters() -> None:
    extraction = extract_assistant_intent(
        "Camera from Canon, minimum budget ₹2000, under ₹8000, rated at least 4.5.",
        groq_client=FailingGroqClient(),
    )

    assert extraction.fallback_used is True
    assert extraction.intent.category == "Camera & Photo"
    assert extraction.intent.brand == "Canon"
    assert extraction.intent.min_budget_override == 2000
    assert extraction.intent.max_budget_override == 8000
    assert extraction.intent.min_rating == 4.5


def test_amazon_catalogue_products_are_recommended_with_unverified_availability(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000001",
        title="Sony Bluetooth Wireless Headphones",
        features=["Bluetooth wireless audio", "Up to 30 hours battery life"],
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            brand="Sony",
            min_rating=4.5,
            max_budget_override=3000,
            include_unverified_availability=True,
        ),
    )

    assert [product.product_id for product in response.products] == [
        "amazon:B000000001"
    ]
    product = response.products[0]
    assert product.catalogue_source == "amazon"
    assert product.availability_status == "unverified"
    assert product.price_is_historical is True
    assert product.rating_count == 100
    assert product.attribute_breakdown["battery_life_priority"] >= 0


def test_recommendations_combine_curated_and_amazon_catalogues(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000010",
        title="Sony Bluetooth Headphones",
        features=["Bluetooth wireless connectivity"],
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
            limit=25,
        ),
    )

    sources = {product.catalogue_source for product in response.products}
    assert sources == {"curated", "amazon"}
    assert len({product.product_id for product in response.products}) == len(
        response.products
    )


def test_combined_catalogues_apply_category_budget_and_brand_filters(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000002",
        title="Sony Wireless Headphones",
        price_inr=2400,
        rating=4.8,
        features=["Bluetooth connectivity"],
    )
    _add_amazon_product(
        db_session,
        asin="B000000003",
        title="Sony Camera",
        category="Camera & Photo",
        price_inr=2200,
        rating=4.9,
    )
    _add_amazon_product(
        db_session,
        asin="B000000004",
        title="Sony Headphones",
        price_inr=1500,
        rating=4.8,
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones & Earbuds",
            min_budget_override=2000,
            max_budget_override=3000,
            brand="Sony",
            min_rating=4.5,
            include_unverified_availability=True,
            limit=25,
        ),
    )

    assert [product.product_id for product in response.products] == [
        "amazon:B000000002"
    ]
    assert all(product.category in {"Headphones", "Headphones & Earbuds"} for product in response.products)
    assert all(2000 <= product.price <= 3000 for product in response.products)


def test_laptop_category_excludes_accessories_and_other_computer_products(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)

    # Create one genuine laptop and three incorrect product matches.
    laptop = _add_amazon_product(
        db_session,
        asin="B000000201",
        title="Acer Aspire 3 Slim Laptop AMD Ryzen 5 8GB RAM 512GB SSD",
        category="Computers & Accessories",
        price_inr=37400,
    )

    headphones = _add_amazon_product(
        db_session,
        asin="B000000202",
        title="AILIHEN C8 Wired Headphones for Chromebook Laptop Computer",
        category="Computers & Accessories",
        price_inr=1800,
    )

    tablet = _add_amazon_product(
        db_session,
        asin="B000000203",
        title="Wacom Intuos Small Bluetooth Graphics Drawing Tablet Chromebook",
        category="Computers & Accessories",
        price_inr=7000,
    )

    external_ssd = _add_amazon_product(
        db_session,
        asin="B000000204",
        title="Seagate Expansion SSD 1TB External Solid State Drive for PC Laptop",
        category="Computers & Accessories",
        price_inr=11000,
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Laptops",
            max_budget_override=50000,
            include_unverified_availability=True,
            limit=25,
        ),
        product_snapshot=[
            laptop,
            headphones,
            tablet,
            external_ssd,
        ],
    )

    returned_ids = {
        product.product_id for product in response.products
    }

    # The genuine laptop must be recommended.
    assert "amazon:B000000201" in returned_ids

    # Other computer-related products must be excluded.
    assert "amazon:B000000202" not in returned_ids
    assert "amazon:B000000203" not in returned_ids
    assert "amazon:B000000204" not in returned_ids

    excluded = {
        product.product_id: product.reason_code
        for product in response.excluded_products
    }

    assert excluded["amazon:B000000202"] == "category_mismatch"
    assert excluded["amazon:B000000203"] == "category_mismatch"
    assert excluded["amazon:B000000204"] == "category_mismatch"

def test_strict_battery_filters_apply_to_both_catalogues(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    curated = {
        "curated-16": 16,
        "curated-20": 20,
        "curated-25": 25,
        "curated-unparseable": "unknown",
        "curated-missing": None,
    }
    for suffix, hours in curated.items():
        _add_curated_battery_product(
            db_session,
            sku=f"SS-BAT-{suffix}",
            battery_life_hours=hours,
        )

    amazon = {
        "B000000101": 16,
        "B000000102": 20,
        "B000000103": 25,
        "B000000104": None,
    }
    for asin, hours in amazon.items():
        features = ["Wireless Bluetooth connectivity"]
        if hours is not None:
            features.append(f"Up to {hours} hours battery life")
        _add_amazon_product(
            db_session,
            asin=asin,
            title=f"TestAudio {asin} Wireless Headphones",
            store=f"TestAudio {asin}",
            features=features,
        )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            max_budget_override=5000,
            include_unverified_availability=True,
            limit=25,
        ),
        required_connectivity="wireless",
        required_product_type="headphones",
        min_battery_life_hours=20,
        require_battery_life_data=True,
    )

    assert {
        product.product_id for product in response.products
    } == {
        "SS-BAT-curated-20",
        "SS-BAT-curated-25",
        "amazon:B000000102",
        "amazon:B000000103",
    }
    excluded = {
        product.product_id: product.reason_code
        for product in response.excluded_products
    }
    assert excluded["SS-BAT-curated-16"] == "battery_life_below_minimum"
    assert excluded["amazon:B000000101"] == "battery_life_below_minimum"
    assert excluded["SS-BAT-curated-unparseable"] == "battery_life_unlisted"
    assert excluded["SS-BAT-curated-missing"] == "battery_life_unlisted"
    assert excluded["amazon:B000000104"] == "battery_life_unlisted"

    listed_only = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
            limit=25,
        ),
        required_connectivity="wireless",
        required_product_type="headphones",
        require_battery_life_data=True,
    )
    listed_ids = {product.product_id for product in listed_only.products}
    assert "SS-BAT-curated-16" in listed_ids
    assert "amazon:B000000101" in listed_ids
    assert "SS-BAT-curated-missing" not in listed_ids
    assert "SS-BAT-curated-unparseable" not in listed_ids


def test_strict_battery_request_returns_no_results_without_relaxing_filters(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    _add_curated_battery_product(
        db_session,
        sku="SS-BAT-NONE",
        battery_life_hours=16,
    )
    _add_amazon_product(
        db_session,
        asin="B000000105",
        title="TestAudio Wireless Headphones",
        features=["Wireless Bluetooth connectivity"],
    )

    response = generate_assistant_response(
        db_session,
        AssistantChatRequest(
            user_id=user_id,
            message=(
                "Wireless headphones under ₹5,000 with at least 20 hours of battery "
                "life. Only recommend products whose battery life is explicitly listed."
            ),
        ),
        groq_client=FailingGroqClient(),
    )

    assert response.recommendations == []
    assert "at least 20 hours of listed battery life" in response.answer
    assert "wireless connectivity" in response.answer
    assert "I didn't relax any filters" in response.answer
    assert any("battery_life_below_minimum" in note for note in response.notes)
    assert not any(
        "Unverified request details" in note
        or "invalid numeric value was not applied" in note
        for note in response.notes
    )
    assert response.fallback_used is True


def test_unparsed_battery_constraint_gets_specific_warning(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)

    response = generate_assistant_response(
        db_session,
        AssistantChatRequest(
            user_id=user_id,
            message="Wireless headphones with at least twenty hours of battery life.",
        ),
        groq_client=FailingGroqClient(),
    )

    assert response.kind == "clarification"
    assert "What minimum battery duration in hours should I use?" in response.answer
    assert any(
        "Unverified request details were not applied: unparsed minimum battery life"
        in note
        for note in response.notes
    )
    assert "An invalid numeric value was not applied." in response.notes


def test_verified_availability_filter_excludes_amazon_and_missing_specs(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000005",
        title="Sony Headphones with unlisted specifications",
        features=["Comfortable listening"],
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=False,
        ),
        required_connectivity="wireless",
    )

    amazon_exclusions = {
        product.product_id: product.reason_code
        for product in response.excluded_products
        if product.catalogue_source == "amazon"
    }
    assert amazon_exclusions["amazon:B000000005"] == "availability_unverified"

    response_with_unverified = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
        ),
        required_connectivity="wireless",
    )
    amazon_exclusions = {
        product.product_id: product.reason_code
        for product in response_with_unverified.excluded_products
        if product.catalogue_source == "amazon"
    }
    assert amazon_exclusions["amazon:B000000005"] == "connectivity_unknown"


def test_follow_up_resolves_both_sources_and_does_not_repeat_products(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000006",
        title="Sony headphones with unknown details",
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
            limit=25,
        ),
    )
    ids = [product.product_id for product in response.products]
    assert "amazon:B000000006" in ids
    assert len(ids) == len(set(ids))

    more = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
            limit=25,
        ),
        excluded_product_ids=ids,
    )
    assert not set(ids).intersection(
        product.product_id for product in more.products
    )

    battery_answer = generate_assistant_response(
        db_session,
        AssistantChatRequest(
            user_id=user_id,
            message="What about battery?",
            conversation_context=AssistantIntent(category="Headphones"),
            previous_product_ids=["amazon:B000000006"],
        ),
        groq_client=FailingGroqClient(),
    )
    assert "does not list battery life" in battery_answer.answer


def test_exact_cross_catalogue_duplicates_are_not_returned_again(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    curated = Product(
        product_id="SS-DUP-001",
        name="TwinBrand Wireless Headphones",
        brand="TwinBrand",
        category="Headphones",
        price=2500,
        rating=4.5,
        description="",
        attributes={"connectivity": "wireless"},
        stock_quantity=5,
        is_active=True,
    )
    db_session.add(curated)
    db_session.commit()
    _add_amazon_product(
        db_session,
        asin="B000000008",
        title="TwinBrand Wireless Headphones",
        store="TwinBrand",
        price_inr=2600,
    )

    first = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
            limit=25,
        ),
    )

    assert [product.product_id for product in first.products] == ["SS-DUP-001"]
    more = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
            limit=25,
        ),
        excluded_product_ids=["SS-DUP-001"],
    )
    assert not more.products
    assert any(
        item.product_id == "amazon:B000000008"
        and item.reason_code == "duplicate_product"
        for item in more.excluded_products
    )


def test_product_details_are_available_for_both_catalogue_sources(
    client,
    db_session: Session,
) -> None:
    seed_products(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000009",
        title="Sony Bluetooth Headphones",
        features=["Bluetooth wireless connectivity"],
    )

    curated = client.get("/api/products/SS-HP-002")
    amazon = client.get("/api/discover/products/B000000009")

    assert curated.status_code == 200
    assert curated.json()["product_id"] == "SS-HP-002"
    assert amazon.status_code == 200
    assert amazon.json()["asin"] == "B000000009"
    assert amazon.json()["features"] == ["Bluetooth wireless connectivity"]


def test_assistant_falls_back_deterministically_and_excludes_unverified_when_requested(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000007",
        title="Sony Bluetooth Headphones",
        features=["Bluetooth wireless connectivity"],
    )

    response = generate_assistant_response(
        db_session,
        AssistantChatRequest(
            user_id=user_id,
            message="Wireless headphones under 5000, currently available.",
        ),
        groq_client=FailingGroqClient(),
    )

    assert response.fallback_used is True
    assert response.intent.availability_requirement == "available"
    assert response.recommendations == []
    assert "verified availability" in response.answer
    assert "Amazon catalogue prices are historical" in response.answer


def test_shop_with_ai_returns_amazon_products_with_historical_disclaimer(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000011",
        title="Sony Bluetooth Headphones",
        features=["Bluetooth wireless connectivity"],
    )

    response = generate_assistant_response(
        db_session,
        AssistantChatRequest(
            user_id=user_id,
            message="Wireless headphones under 5000.",
            limit=25,
        ),
        groq_client=FailingGroqClient(),
    )

    amazon_recommendations = [
        product
        for product in response.recommendations
        if product.catalogue_source == "amazon"
    ]
    assert amazon_recommendations
    assert all(product.availability_status == "unverified" for product in amazon_recommendations)
    assert "historical USD-to-INR conversions" in response.answer
    assert "not current Indian retail prices" in response.answer


def test_wireless_filter_uses_title_and_excludes_wired_and_unknown_products(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    wired = _add_amazon_product(
        db_session,
        asin="B000000012",
        title="iRAG J01 Kids Headphones Wired",
    )
    wireless = _add_amazon_product(
        db_session,
        asin="B000000013",
        title="Sony Wireless Over-Ear Headphones",
        features=["Bluetooth wireless audio"],
    )
    unknown = _add_amazon_product(
        db_session,
        asin="B000000014",
        title="Sony Over-Ear Headphones",
    )
    contradictory = _add_amazon_product(
        db_session,
        asin="B000000028",
        title="Sony Wireless Over-Ear Headphones",
        features=["Wireless connectivity is not supported"],
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            max_budget_override=5000,
            include_unverified_availability=True,
        ),
        required_connectivity="wireless",
        required_product_type="headphones",
        product_snapshot=[wired, wireless, unknown, contradictory],
    )

    assert [product.product_id for product in response.products] == [
        "amazon:B000000013"
    ]
    excluded = {item.product_id: item.reason_code for item in response.excluded_products}
    assert excluded["amazon:B000000012"] == "connectivity_mismatch"
    assert excluded["amazon:B000000014"] == "connectivity_unknown"
    assert excluded["amazon:B000000028"] == "connectivity_unknown"


def test_wired_filter_excludes_wireless_products_and_unknown_connectivity(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    wired = _add_amazon_product(
        db_session,
        asin="B000000015",
        title="Sony Wired Over-Ear Headphones",
    )
    wireless = _add_amazon_product(
        db_session,
        asin="B000000016",
        title="Sony Bluetooth Headphones",
    )
    unknown = _add_amazon_product(
        db_session,
        asin="B000000017",
        title="Sony Over-Ear Headphones",
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            max_budget_override=5000,
            include_unverified_availability=True,
        ),
        required_connectivity="wired",
        required_product_type="headphones",
        product_snapshot=[wired, wireless, unknown],
    )

    assert [product.product_id for product in response.products] == [
        "amazon:B000000015"
    ]
    excluded = {item.product_id: item.reason_code for item in response.excluded_products}
    assert excluded["amazon:B000000016"] == "connectivity_mismatch"
    assert excluded["amazon:B000000017"] == "connectivity_unknown"


def test_headphone_product_types_exclude_sleep_products_and_accessories(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    regular = _add_amazon_product(
        db_session,
        asin="B000000018",
        title="Sony Wireless Over-Ear Headphones",
        features=["Bluetooth audio"],
    )
    earbuds = _add_amazon_product(
        db_session,
        asin="B000000019",
        title="Sony Wireless Earbuds",
        features=["Bluetooth audio"],
    )
    sleep = _add_amazon_product(
        db_session,
        asin="B000000020",
        title="LIGHTIMETUNNEL Bluetooth Sleep Headphones Sleep Mask",
        features=["Bluetooth audio"],
    )
    accessory = _add_amazon_product(
        db_session,
        asin="B000000021",
        title="Replacement Ear Pads for Sony Headphones",
    )

    regular_response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
        ),
        required_product_type="headphones",
        product_snapshot=[regular, earbuds, sleep, accessory],
    )
    assert [product.product_id for product in regular_response.products] == [
        "amazon:B000000018"
    ]
    excluded = {
        item.product_id: item.reason_code for item in regular_response.excluded_products
    }
    assert excluded["amazon:B000000019"] == "product_type_mismatch"
    assert excluded["amazon:B000000020"] == "product_type_mismatch"
    assert excluded["amazon:B000000021"] == "headphone_accessory"

    sleep_response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            include_unverified_availability=True,
        ),
        required_product_type="sleep_headphones",
        product_snapshot=[sleep],
    )
    assert [product.product_id for product in sleep_response.products] == [
        "amazon:B000000020"
    ]


def test_battery_data_is_prioritized_and_missing_data_is_explained(
    db_session: Session,
) -> None:
    user_id = _create_profile(db_session)
    verified_battery = _add_amazon_product(
        db_session,
        asin="B000000022",
        title="Sony Wireless Over-Ear Headphones with 40-hour Battery Life",
        price_inr=2500,
        rating=4.5,
        features=["Bluetooth audio"],
    )
    missing_battery = _add_amazon_product(
        db_session,
        asin="B000000023",
        title="Sony Wireless Over-Ear Headphones Pro",
        price_inr=2500,
        rating=4.5,
        features=["Bluetooth audio"],
    )

    response = generate_recommendations(
        db_session,
        RecommendationRequest(
            user_id=user_id,
            category="Headphones",
            max_budget_override=5000,
            include_unverified_availability=True,
        ),
        preference_overrides={"battery_life_priority": 80},
        required_connectivity="wireless",
        required_product_type="headphones",
        product_snapshot=[verified_battery, missing_battery],
    )

    assert [product.product_id for product in response.products] == [
        "amazon:B000000022",
        "amazon:B000000023",
    ]
    assert "Catalogue lists 40 hours of battery life." in response.products[0].reasons
    assert "Battery life is not listed in catalogue data." in response.products[1].reasons
    assert response.products[0].score > response.products[1].score


def test_shop_with_ai_enforces_budget_and_explains_verified_attributes(
    db_session: Session,
) -> None:
    seed_products(db_session)
    user_id = _create_profile(db_session)
    _add_amazon_product(
        db_session,
        asin="B000000024",
        title="Sony Wireless Over-Ear Headphones",
        price_inr=4500,
        rating=4.7,
        features=["Bluetooth audio", "Up to 35 hours of battery life"],
    )
    _add_amazon_product(
        db_session,
        asin="B000000025",
        title="Sony Wireless Over-Ear Headphones Lite",
        price_inr=4200,
        rating=4.6,
        features=["Bluetooth audio"],
    )
    _add_amazon_product(
        db_session,
        asin="B000000026",
        title="Sony Wired Over-Ear Headphones",
        price_inr=3000,
        rating=4.8,
    )
    _add_amazon_product(
        db_session,
        asin="B000000027",
        title="Sony Bluetooth Earbuds",
        price_inr=2000,
        rating=4.9,
        features=["Bluetooth audio", "50 hours battery life"],
    )

    response = generate_assistant_response(
        db_session,
        AssistantChatRequest(
            user_id=user_id,
            message=(
                "I need wireless headphones under ₹5,000 with good battery life "
                "and rating at least 4.5."
            ),
            limit=25,
        ),
        groq_client=FailingGroqClient(),
    )

    assert response.intent.product_type_requirement == "headphones"
    assert response.intent.connectivity_requirement == "wireless"
    assert response.intent.battery_life_priority is not None
    assert response.intent.max_budget_override == 5000
    assert response.intent.min_budget_override is None
    assert response.recommendations
    assert all(product.price <= 5000 for product in response.recommendations)
    assert all(product.rating >= 4.5 for product in response.recommendations)
    for product in response.recommendations:
        assert "Catalogue identifies its connection as wireless." in product.reasons
        assert "Catalogue identifies this as conventional headphones." in product.reasons
        assert any(
            "within your maximum budget of ₹5,000" in reason
            for reason in product.reasons
        )
        assert any("meets your 4.5/5 minimum" in reason for reason in product.reasons)
    assert all(
        product.product_id not in {"amazon:B000000026", "amazon:B000000027"}
        for product in response.recommendations
    )
