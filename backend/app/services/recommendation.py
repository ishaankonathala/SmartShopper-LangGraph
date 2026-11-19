"""Deterministic, explainable recommendation engine."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.dna_defaults import PREFERENCE_DIMENSIONS
from app.models.amazon_catalogue_product import AmazonCatalogueProduct
from app.models.product import Product
from app.models.shopper_dna import ShopperDNA
from app.models.user_profile import UserProfile
from app.schemas.assistant import ConnectivityRequirement, ProductTypeRequirement
from app.schemas.assistant import AssistantIntent
from app.schemas.recommendation import (
    AvailabilityStatus,
    ExcludedProduct,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedProduct,
)

AMAZON_PRODUCT_ID_PREFIX = "amazon:"
_LAPTOP_TITLE_PATTERN = re.compile(r"\b(?:laptop|notebook|chromebook|macbook)\b", re.I)
_LAPTOP_ACCESSORY_PATTERN = re.compile(
    r"\b(?:case|sleeve|skin|screen protector|charger|charging cable|"
    r"adapter|hub|dock|docking station|stand|replacement battery|"
    r"keyboard cover|keyboard|mouse|mice|backpack|bag|"
    r"ethernet cable|usb cable|usb c cable|displayport cable|hdmi cable|"
    r"webcam|web camera|external hard drive|hard drive|"
    r"external ssd|portable ssd|solid state drive|"
    r"cooling pad|laptop cooler|graphics tablet|drawing tablet|"
    r"headphones|earbuds|bluetooth adapter|usb receiver)\b",
    re.I,
)

_SMART_WEARABLE_TITLE_PATTERN = re.compile(
    r"\b(?:smart\s?watch|fitness tracker|activity tracker|smart band)\b",
    re.I,
)
_SMART_WEARABLE_ACCESSORY_PATTERN = re.compile(
    r"\b(?:replacement band|watch band|watch strap|screen protector|charging cable)\b",
    re.I,
)
_HEADPHONE_ACCESSORY_PATTERN = re.compile(
    r"\b(?:headphone|earphone|earbud)s?\s+"
    r"(?:accessories|case|stand|holder|mount|cable|adapter|cover|pads?|cushions?)\b|"
    r"\b(?:replacement|spare|protective)\s+"
    r"(?:headphone|earphone|earbud)?\s*(?:ear\s*)?"
    r"(?:pads?|cushions?|tips|cases?)\b|"
    r"\b(?:ear\s*pads?|ear\s*cushions?|headphone accessories)\b|"
    r"\b(?:case|stand|holder|mount|cable|adapter)\s+for\s+"
    r"(?:headphone|earphone|earbud)s?\b",
    re.I,
)
_SLEEP_HEADPHONE_PATTERN = re.compile(
    r"\b(?:sleep(?:ing)?\s+(?:headphones?|headsets?|earbuds?|earphones?)|"
    r"(?:headphones?|headsets?|earbuds?)\s+for\s+sleep)\b",
    re.I,
)
_EARBUD_PATTERN = re.compile(r"\b(?:ear[ -]?buds?|earphones?|in[- ]ear)\b", re.I)
_HEADPHONE_PATTERN = re.compile(r"\b(?:headphones?|headsets?)\b", re.I)
_CHILDREN_HEADPHONE_TITLE_PATTERN = re.compile(
    r"\b(?:kids?|children(?:['’]s)?|child(?:ren)?|toddlers?)\b",
    re.I,
)
_CHILD_AUDIENCE_PATTERN = re.compile(
    r"\b(?:kids?|children(?:['’]s)?|child(?:ren)?|toddlers?|son|daughter)\b",
    re.I,
)
_AUDIO_REQUEST_PATTERN = re.compile(
    r"\b(?:headphones?|headsets?|earbuds?|earphones?)\b",
    re.I,
)
_CHILD_REQUEST_NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|avoid|without|exclude|excluding|except|don't|dont)\b",
    re.I,
)
_BATTERY_PATTERNS = (
    re.compile(
        r"\b(?:up to\s+)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*"
        r"(?:of\s+)?(?:battery(?: life)?|playback(?: time)?|play\s*time|"
        r"playtime|talk time)\b",
        re.I,
    ),
    re.compile(
        r"\bbattery(?: life)?(?:\s+(?:is|of|up to))?\s*[:\-]?\s*"
        r"(?:up to\s+)?"
        r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b",
        re.I,
    ),
    re.compile(
        r"\b(\d+(?:\.\d+)?)\s*[- ]?hours?\s+(?:of\s+)?battery(?: life)?\b",
        re.I,
    ),
)
_WIRELESS_CONNECTIVITY_PATTERN = re.compile(
    r"\b(?:wireless|bluetooth|wi-?fi)\b",
    re.I,
)
_WIRED_CONNECTIVITY_PATTERN = re.compile(r"\b(?:wired|cabled)\b", re.I)
_CONNECTIVITY_NEGATION_PREFIX_PATTERN = re.compile(
    r"\b(?:not|no|non|without|unsupported|lacks?|does not support|doesn't support)"
    r"\s+(?:\w+\s+){0,2}$",
    re.I,
)
_CONNECTIVITY_NEGATION_SUFFIX_PATTERN = re.compile(
    r"^\s*(?::\s*)?(?:\w+\s+){0,2}(?:[:\-]\s*)?(?:is\s+)?"
    r"(?:not supported|unsupported|not included|not available)\b",
    re.I,
)


def _canonical_recommendation_id(product_id: str) -> str:
    if product_id.casefold().startswith(AMAZON_PRODUCT_ID_PREFIX):
        asin = product_id[len(AMAZON_PRODUCT_ID_PREFIX) :]
        return f"{AMAZON_PRODUCT_ID_PREFIX}{asin.upper()}"
    return product_id


def _classify_audio_product(title: str, catalogue_text: str) -> str | None:
    normalized_title = title.casefold()
    normalized_text = catalogue_text.casefold()
    title_has_audio_product = bool(
        _HEADPHONE_PATTERN.search(normalized_title)
        or _EARBUD_PATTERN.search(normalized_title)
    )
    if _HEADPHONE_ACCESSORY_PATTERN.search(normalized_title):
        return "accessory"
    if (
        _SLEEP_HEADPHONE_PATTERN.search(normalized_text)
        or (
            re.search(r"\bsleep\s+masks?\b", normalized_text)
            and title_has_audio_product
        )
    ) and title_has_audio_product:
        return "sleep_headphones"
    if re.search(r"\bsleep\s+masks?\b", normalized_title) and not title_has_audio_product:
        return "accessory"
    if _EARBUD_PATTERN.search(normalized_text):
        return "earbuds"
    if _HEADPHONE_PATTERN.search(normalized_text):
        return "headphones"
    return None


def _candidate_product_type(product: RecommendationCandidate) -> str | None:
    attributes = product.attributes if isinstance(product.attributes, dict) else {}
    product_type = attributes.get("product_type")
    if isinstance(product_type, str) and product_type in {
        "headphones",
        "earbuds",
        "sleep_headphones",
        "accessory",
    }:
        return product_type
    attribute_text = " ".join(str(value) for value in attributes.values())
    return _classify_audio_product(
        product.name,
        f"{product.name} {product.description} {attribute_text}",
    )


def explicitly_requests_children_headphones(message: str) -> bool:
    """Recognize a clear child-audience request without inferring age suitability."""
    audience_matches = list(_CHILD_AUDIENCE_PATTERN.finditer(message))
    audio_matches = list(_AUDIO_REQUEST_PATTERN.finditer(message))
    for audience in audience_matches:
        for audio in audio_matches:
            if abs(audience.start() - audio.start()) > 80:
                continue
            start = max(0, min(audience.start(), audio.start()) - 24)
            end = min(len(message), max(audience.end(), audio.end()) + 24)
            if _CHILD_REQUEST_NEGATION_PATTERN.search(message[start:end]):
                continue
            return True
    return False


def _battery_life_hours(product: RecommendationCandidate) -> float | None:
    attributes = product.attributes if isinstance(product.attributes, dict) else {}
    value = attributes.get("battery_life_hours")
    try:
        hours = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return hours if math.isfinite(hours) and hours > 0 else None


def candidate_product_type(product: RecommendationCandidate) -> str | None:
    """Return the same inferred product type used by strict recommendation filters."""
    return _candidate_product_type(product)


def candidate_battery_life_hours(
    product: RecommendationCandidate,
) -> float | None:
    """Return a positive listed battery duration, or None when unavailable."""
    return _battery_life_hours(product)


def _amazon_connectivity(text: str) -> str | None:
    positive: set[str] = set()
    negated: set[str] = set()
    for label, pattern in (
        ("wireless", _WIRELESS_CONNECTIVITY_PATTERN),
        ("wired", _WIRED_CONNECTIVITY_PATTERN),
    ):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 48) : match.start()]
            suffix = text[match.end() : match.end() + 48]
            is_negated = bool(
                _CONNECTIVITY_NEGATION_PREFIX_PATTERN.search(prefix)
                or _CONNECTIVITY_NEGATION_SUFFIX_PATTERN.match(suffix)
            )
            if is_negated and re.search(r"\bnot\s+only\s*$", prefix, re.I):
                is_negated = False
            (negated if is_negated else positive).add(label)
    if positive.intersection(negated) or len(positive) != 1:
        return None
    return next(iter(positive))


@dataclass(frozen=True)
class RecommendationCandidate:
    product_id: str
    catalogue_source: Literal["curated", "amazon"]
    name: str
    brand: str | None
    category: str
    price: float
    rating: float
    image_url: str | None
    description: str
    attributes: dict[str, Any]
    is_active: bool
    availability_status: AvailabilityStatus
    rating_count: int | None = None
    price_is_historical: bool = False


def _amazon_specifications(product: AmazonCatalogueProduct) -> dict[str, Any]:
    text = " ".join(
        [product.title, *(product.features or []), *(product.description or [])]
    )
    attributes: dict[str, Any] = {}
    connectivity = _amazon_connectivity(text)
    if connectivity is not None:
        attributes["connectivity"] = connectivity

    for pattern in _BATTERY_PATTERNS:
        match = pattern.search(text)
        if match:
            hours = float(match.group(1))
            if math.isfinite(hours) and hours > 0:
                attributes["battery_life_hours"] = hours
                break
    product_type = _classify_audio_product(product.title, text)
    if product_type is not None:
        attributes["product_type"] = product_type
    return attributes


def _as_candidate(
    product: Product | AmazonCatalogueProduct | RecommendationCandidate,
) -> RecommendationCandidate:
    if isinstance(product, RecommendationCandidate):
        return product
    if isinstance(product, Product):
        available = bool(product.is_active and product.stock_quantity > 0)
        return RecommendationCandidate(
            product_id=product.product_id,
            catalogue_source="curated",
            name=product.name,
            brand=product.brand,
            category=product.category,
            price=float(product.price),
            rating=float(product.rating),
            image_url=product.image_url,
            description=product.description,
            attributes=product.attributes if isinstance(product.attributes, dict) else {},
            is_active=product.is_active,
            availability_status="available" if available else "unavailable",
        )

    return RecommendationCandidate(
        product_id=f"{AMAZON_PRODUCT_ID_PREFIX}{product.asin}",
        catalogue_source="amazon",
        name=product.title,
        brand=product.store,
        category=product.category,
        price=float(product.price_inr),
        rating=float(product.rating),
        image_url=product.image_url,
        description="\n".join(product.description or []),
        attributes=_amazon_specifications(product),
        is_active=True,
        availability_status="unverified",
        rating_count=product.rating_count,
        price_is_historical=True,
    )


def load_recommendation_products(
    db: Session,
    product_ids: Sequence[str],
) -> list[RecommendationCandidate]:
    """Resolve source-qualified recommendation identities in their original order."""
    if not product_ids:
        return []

    amazon_by_id: dict[str, RecommendationCandidate] = {}
    curated_by_id: dict[str, RecommendationCandidate] = {}
    canonical_ids = [_canonical_recommendation_id(product_id) for product_id in product_ids]
    amazon_asins = [
        product_id[len(AMAZON_PRODUCT_ID_PREFIX) :]
        for product_id in canonical_ids
        if product_id.startswith(AMAZON_PRODUCT_ID_PREFIX)
    ]
    curated_ids = [
        product_id
        for product_id in canonical_ids
        if not product_id.startswith(AMAZON_PRODUCT_ID_PREFIX)
    ]
    if amazon_asins:
        amazon_products = db.scalars(
            select(AmazonCatalogueProduct).where(
                AmazonCatalogueProduct.asin.in_(amazon_asins)
            )
        ).all()
        amazon_by_id = {
            candidate.product_id: candidate
            for candidate in (_as_candidate(product) for product in amazon_products)
        }
    if curated_ids:
        curated_products = db.scalars(
            select(Product).where(Product.product_id.in_(curated_ids))
        ).all()
        curated_by_id = {
            candidate.product_id: candidate
            for candidate in (_as_candidate(product) for product in curated_products)
        }
    products_by_id = {**curated_by_id, **amazon_by_id}
    return [products_by_id[product_id] for product_id in canonical_ids if product_id in products_by_id]


DIMENSION_WEIGHTS: dict[str, float] = {
    "comfort_priority": 0.25,
    "brand_loyalty": 0.20,
    "price_sensitivity": 0.20,
    "quality_preference": 0.20,
    "portability_preference": 0.10,
    "battery_life_priority": 0.05,
}

DIMENSION_ORDER: tuple[str, ...] = tuple(PREFERENCE_DIMENSIONS)
PreferenceOverrideDimension = Literal[
    "comfort_priority",
    "brand_loyalty",
    "battery_life_priority",
    "portability_preference",
]


def preference_overrides_from_intent(
    intent: AssistantIntent,
) -> dict[PreferenceOverrideDimension, int]:
    """Translate validated assistant priorities into recommendation overrides."""
    overrides: dict[PreferenceOverrideDimension, int] = {}
    if intent.comfort_priority is not None:
        overrides["comfort_priority"] = intent.comfort_priority
    if intent.brand_loyalty is not None:
        overrides["brand_loyalty"] = intent.brand_loyalty
    if intent.battery_life_priority is not None:
        overrides["battery_life_priority"] = intent.battery_life_priority
    if intent.portability_priority is not None:
        overrides["portability_preference"] = intent.portability_priority
    return overrides


def _clamp(value: float, *, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _effective_dna_value(
    dna: ShopperDNA | None,
    dimension: str,
    preference_overrides: Mapping[PreferenceOverrideDimension, int],
) -> float:
    if dimension in preference_overrides:
        return _clamp(float(preference_overrides[dimension]))

    if dna is None:
        return 50.0

    inferred = dna.inferred_preferences.get(dimension, 50)
    confidence = dna.confidence_scores.get(dimension, 0)
    source = dna.preference_sources.get(dimension, "default")
    if source == "default":
        confidence = 0
    return _clamp(50 + (inferred - 50) * (confidence / 100.0))


def _matches_disliked_attributes(
    product: RecommendationCandidate,
    disliked: list[str],
) -> bool:
    if not disliked:
        return False

    attributes = product.attributes if isinstance(product.attributes, dict) else {}
    normalized_dislikes = {item.strip().lower() for item in disliked if str(item).strip()}
    if not normalized_dislikes:
        return False

    for key, value in attributes.items():
        key_text = str(key).lower()
        if key_text in normalized_dislikes:
            return True
        if isinstance(value, str):
            value_text = value.lower()
            if value_text in normalized_dislikes or any(
                dislike in value_text for dislike in normalized_dislikes
            ):
                return True
        else:
            value_text = str(value).lower()
            if value_text in normalized_dislikes or any(
                dislike in value_text for dislike in normalized_dislikes
            ):
                return True
    return False


def _product_brand_match(
    product: RecommendationCandidate,
    preferred_brands: list[str],
) -> bool:
    if not preferred_brands:
        return False
    preferred = {brand.strip().lower() for brand in preferred_brands if str(brand).strip()}
    return bool(product.brand and product.brand.strip().lower() in preferred)


def _feature_value_for_dimension(
    product: RecommendationCandidate,
    dimension: str,
    *,
    preferred_brands: list[str] | None = None,
    max_budget: float | None = None,
    battery_max_hours: float | None = None,
) -> float | None:
    attributes = product.attributes if isinstance(product.attributes, dict) else {}

    if dimension == "comfort_priority":
        value = attributes.get("comfort_score")
        if value is None:
            return None
        try:
            return _clamp(float(value))
        except (TypeError, ValueError):
            return None

    if dimension == "brand_loyalty":
        if not preferred_brands or not product.brand:
            return None
        return 100.0 if _product_brand_match(product, preferred_brands) else 0.0

    if dimension == "price_sensitivity":
        if max_budget is None or max_budget <= 0:
            return None
        if product.price <= 0:
            return 100.0
        ratio = product.price / max_budget
        return _clamp(100.0 - (ratio * 100.0))

    if dimension == "quality_preference":
        build_quality = attributes.get("build_quality")
        if build_quality is not None:
            try:
                return _clamp(float(build_quality))
            except (TypeError, ValueError):
                pass
        rating = product.rating
        if rating is None:
            return None
        try:
            return _clamp((float(rating) / 5.0) * 100.0)
        except (TypeError, ValueError):
            return None

    if dimension == "portability_preference":
        value = attributes.get("portability")
        if value is None:
            return None
        try:
            return _clamp(float(value))
        except (TypeError, ValueError):
            return None

    if dimension == "battery_life_priority":
        value = attributes.get("battery_life_hours")
        if value is None or battery_max_hours is None or battery_max_hours <= 0:
            return None
        try:
            hours = float(value)
            if not math.isfinite(hours) or hours <= 0:
                return None
            return _clamp((hours / battery_max_hours) * 100.0)
        except (TypeError, ValueError):
            return None

    return None


def _matches_category(
    product: RecommendationCandidate,
    requested_category: str,
) -> bool:
    product_category = product.category.strip().casefold()
    requested = requested_category.strip().casefold()
    if product_category == requested:
        return True

    if {product_category, requested} == {
        "headphones",
        "headphones & earbuds",
    }:
        return True

    if product.catalogue_source != "amazon":
        return False
    if requested == "laptops" and product_category == "computers & accessories":
        return bool(
            _LAPTOP_TITLE_PATTERN.search(product.name)
            and not _LAPTOP_ACCESSORY_PATTERN.search(product.name)
        )
    if requested == "smartwatches" and product_category == "wearable technology":
        return bool(
            _SMART_WEARABLE_TITLE_PATTERN.search(product.name)
            and not _SMART_WEARABLE_ACCESSORY_PATTERN.search(product.name)
        )
    return False


def _candidate_identity(product: RecommendationCandidate) -> tuple[str, str] | None:
    if not product.brand:
        return None
    normalized_name = " ".join(
        re.sub(r"[^a-z0-9]+", " ", product.name.casefold()).split()
    )
    normalized_brand = " ".join(
        re.sub(r"[^a-z0-9]+", " ", product.brand.casefold()).split()
    )
    if not normalized_name or not normalized_brand:
        return None
    return normalized_brand, normalized_name


def _compute_exclusion_reason(
    product: RecommendationCandidate,
    *,
    request: RecommendationRequest,
    profile: UserProfile,
    require_available: bool,
    required_connectivity: ConnectivityRequirement | None,
    required_product_type: ProductTypeRequirement | None,
    min_battery_life_hours: float | None,
    require_battery_life_data: bool,
) -> ExcludedProduct | None:
    if not product.is_active:
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="inactive_product",
            message="Product is inactive and excluded from recommendations.",
            catalogue_source=product.catalogue_source,
        )

    if (
        require_available
        and product.availability_status != "available"
        and not (
            product.catalogue_source == "amazon"
            and product.availability_status == "unverified"
            and request.include_unverified_availability
        )
    ):
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code=(
                "availability_unverified"
                if product.availability_status == "unverified"
                else "out_of_stock"
            ),
            message=(
                "Current availability could not be verified."
                if product.availability_status == "unverified"
                else "Product is out of stock and excluded from recommendations."
            ),
            catalogue_source=product.catalogue_source,
        )

    effective_budget = (
        request.max_budget_override if request.max_budget_override is not None else profile.max_budget
    )
    if product.price > effective_budget:
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="budget_exceeded",
            message=(
                f"Product price {product.price} exceeds the effective budget of {effective_budget}."
            ),
            catalogue_source=product.catalogue_source,
        )

    effective_min_budget = max(
        (
            value
            for value in (profile.min_budget, request.min_budget_override)
            if value is not None
        ),
        default=None,
    )
    if effective_min_budget is not None and product.price < effective_min_budget:
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="below_min_budget",
            message=(
                f"Product price {product.price} is below the minimum budget of {effective_min_budget}."
            ),
            catalogue_source=product.catalogue_source,
        )

    requested_category = request.effective_category
    if requested_category and not _matches_category(product, requested_category):
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="category_mismatch",
            message=(
                f"Product category '{product.category}' does not match the requested category "
                f"'{requested_category}'."
            ),
            catalogue_source=product.catalogue_source,
        )

    if requested_category and requested_category.strip().casefold() in {
        "headphones",
        "headphones & earbuds",
    }:
        product_type = _candidate_product_type(product)
        if product_type == "accessory":
            return ExcludedProduct(
                product_id=product.product_id,
                name=product.name,
                reason_code="headphone_accessory",
                message="A headphone accessory is not a headphone product.",
                catalogue_source=product.catalogue_source,
            )
        if product_type is None:
            return ExcludedProduct(
                product_id=product.product_id,
                name=product.name,
                reason_code="product_type_unknown",
                message="Product type is not clear enough to verify headphone relevance.",
                catalogue_source=product.catalogue_source,
            )
        if product_type == "sleep_headphones" and required_product_type != "sleep_headphones":
            return ExcludedProduct(
                product_id=product.product_id,
                name=product.name,
                reason_code="product_type_mismatch",
                message="Sleep headphones do not match a request for regular headphones.",
                catalogue_source=product.catalogue_source,
            )
        if required_product_type is not None and product_type != required_product_type:
            return ExcludedProduct(
                product_id=product.product_id,
                name=product.name,
                reason_code="product_type_mismatch",
                message="Product type does not match the requested headphone type.",
                catalogue_source=product.catalogue_source,
            )

    if request.brand and (
        not product.brand
        or product.brand.strip().casefold() != request.brand.strip().casefold()
    ):
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="brand_mismatch",
            message="Product brand does not match the requested brand.",
            catalogue_source=product.catalogue_source,
        )

    if request.min_rating is not None and product.rating < request.min_rating:
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="rating_below_minimum",
            message="Product rating is below the requested minimum.",
            catalogue_source=product.catalogue_source,
        )

    if required_connectivity is not None:
        attributes = product.attributes if isinstance(product.attributes, dict) else {}
        connectivity = attributes.get("connectivity")
        if connectivity not in {"wireless", "wired"}:
            return ExcludedProduct(
                product_id=product.product_id,
                name=product.name,
                reason_code="connectivity_unknown",
                message="Product connectivity is unknown and cannot satisfy the requested filter.",
                catalogue_source=product.catalogue_source,
            )
        if connectivity != required_connectivity:
            return ExcludedProduct(
                product_id=product.product_id,
                name=product.name,
                reason_code="connectivity_mismatch",
                message="Product connectivity does not match the requested filter.",
                catalogue_source=product.catalogue_source,
            )

    battery_life_hours = _battery_life_hours(product)
    if min_battery_life_hours is not None and battery_life_hours is None:
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="battery_life_unlisted",
            message="A verifiable battery duration is not listed.",
            catalogue_source=product.catalogue_source,
        )
    if (
        min_battery_life_hours is not None
        and battery_life_hours is not None
        and battery_life_hours < min_battery_life_hours
    ):
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="battery_life_below_minimum",
            message="Listed battery duration is below the requested minimum.",
            catalogue_source=product.catalogue_source,
        )
    if require_battery_life_data and battery_life_hours is None:
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="battery_life_unlisted",
            message="A verifiable battery duration is not listed.",
            catalogue_source=product.catalogue_source,
        )

    if _matches_disliked_attributes(product, profile.disliked_attributes):
        return ExcludedProduct(
            product_id=product.product_id,
            name=product.name,
            reason_code="disliked_attribute",
            message="Product attributes match a disliked attribute or value in the profile.",
            catalogue_source=product.catalogue_source,
        )

    return None


def _score_product(
    product: RecommendationCandidate,
    *,
    profile: UserProfile,
    dna: ShopperDNA | None,
    max_budget: float,
    min_budget: float | None,
    min_rating: float | None,
    requested_category: str | None,
    required_connectivity: ConnectivityRequirement | None,
    required_product_type: ProductTypeRequirement | None,
    battery_max_hours: float | None,
    include_explanations: bool,
    preference_overrides: Mapping[PreferenceOverrideDimension, int],
) -> RecommendedProduct:
    reasons: list[str] = []
    attribute_breakdown: dict[str, float] = {}
    score_breakdown: dict[str, float] = {}
    active_weights: dict[str, float] = {}
    active_confidences: dict[str, float] = {}
    preference_reasons: dict[str, str] = {}
    priority_reason_dimensions = _priority_reason_dimensions(preference_overrides)
    preferred_reason_dimensions = set(priority_reason_dimensions)
    preferred_brands = list(profile.preferred_brands or [])
    attributes = product.attributes if isinstance(product.attributes, dict) else {}
    brand_match = _product_brand_match(product, preferred_brands)

    if brand_match:
        reasons.append(f"Matches your preferred brand {product.brand}.")
    if min_budget is not None and min_budget > 0:
        reasons.append(
            f"Listed price ₹{product.price:,.0f} is within your "
            f"₹{min_budget:,.0f}–₹{max_budget:,.0f} budget."
        )
    else:
        reasons.append(
            f"Listed price ₹{product.price:,.0f} is within your maximum "
            f"budget of ₹{max_budget:,.0f}."
        )
    if required_connectivity is not None:
        reasons.append(
            f"Catalogue identifies its connection as {required_connectivity}."
        )
    if required_product_type is not None:
        product_type_label = {
            "headphones": "conventional headphones",
            "earbuds": "earbuds",
            "sleep_headphones": "sleep headphones",
        }[required_product_type]
        reasons.append(f"Catalogue identifies this as {product_type_label}.")
    if min_rating is not None:
        reasons.append(
            f"Catalogue rating {product.rating:.1f}/5 meets your "
            f"{min_rating:g}/5 minimum."
        )
    if product.catalogue_source == "amazon":
        reasons.extend(
            [
                "The converted price is illustrative historical Amazon catalogue data, not a current Indian retail price.",
                "Current product availability is unverified.",
            ]
        )

    for dimension in DIMENSION_ORDER:
        feature_value = _feature_value_for_dimension(
            product,
            dimension,
            preferred_brands=preferred_brands,
            max_budget=max_budget,
            battery_max_hours=battery_max_hours,
        )
        if (
            feature_value is None
            and dimension == "battery_life_priority"
            and preference_overrides.get(dimension, 0) >= 60
        ):
            feature_value = 0.0
        if feature_value is None:
            continue

        weight = DIMENSION_WEIGHTS[dimension]
        active_weights[dimension] = weight
        attribute_breakdown[dimension] = round(_clamp(feature_value), 2)

        effective_value = _effective_dna_value(dna, dimension, preference_overrides)
        contribution = (feature_value / 100.0) * (effective_value / 100.0) * 100.0
        score_breakdown[dimension] = round(contribution * weight, 2)

        if dimension in preference_overrides:
            confidence_value = 100.0
        else:
            source = (dna.preference_sources.get(dimension, "default") if dna else "default")
            if source == "default":
                confidence_value = 0.0
            else:
                confidence_value = float(dna.confidence_scores.get(dimension, 0)) if dna else 0.0
        active_confidences[dimension] = confidence_value

        if priority_reason_dimensions and dimension not in preferred_reason_dimensions:
            continue
        if dimension == "comfort_priority" and feature_value >= 75:
            preference_reasons[dimension] = (
                "Strong comfort score matches your comfort preference."
            )
        if dimension == "quality_preference" and feature_value >= 75:
            build_quality = attributes.get("build_quality")
            try:
                build_quality_score = (
                    None if isinstance(build_quality, bool) else float(build_quality)
                )
            except (OverflowError, TypeError, ValueError):
                build_quality_score = None
            if (
                build_quality_score is not None
                and math.isfinite(build_quality_score)
            ):
                preference_reasons[dimension] = (
                    f"Catalogue build-quality score {_clamp(build_quality_score):g}/100 "
                    "is listed."
                )
            else:
                preference_reasons[dimension] = (
                    f"Catalogue rating {product.rating:.1f}/5 is listed; "
                    "build-quality data is unavailable."
                )
        if dimension == "portability_preference" and feature_value >= 75:
            preference_reasons[dimension] = (
                "High portability aligns with your portability preference."
            )
        if dimension == "battery_life_priority":
            battery_hours = _battery_life_hours(product)
            if battery_hours is not None:
                preference_reasons[dimension] = (
                    f"Catalogue lists {battery_hours:g} hours of battery life."
                )

    if (
        "comfort_priority" in preference_overrides
        and "comfort_priority" not in attribute_breakdown
    ):
        preference_reasons["comfort_priority"] = (
            "Catalogue comfort score is unavailable."
        )
    if (
        "portability_preference" in preference_overrides
        and "portability_preference" not in attribute_breakdown
    ):
        preference_reasons["portability_preference"] = (
            "Catalogue portability data is unavailable."
        )

    reason_order = priority_reason_dimensions or list(DIMENSION_ORDER)
    reasons.extend(
        preference_reasons[dimension]
        for dimension in reason_order
        if dimension in preference_reasons
    )
    if "battery_life_priority" in preference_overrides and _battery_life_hours(product) is None:
        reasons.append("Battery life is not listed in catalogue data.")
    if requested_category and _matches_category(product, requested_category):
        if required_product_type is None:
            reasons.append(f"Matches your requested category '{requested_category}'.")

    if not active_weights:
        final_score = 0.0
        final_confidence = 0.0
    else:
        total_weight = sum(active_weights.values())
        final_score = round(sum(score_breakdown.values()) / total_weight, 2)
        final_confidence = round(
            sum(active_confidences.get(dim, 0.0) * weight for dim, weight in active_weights.items())
            / total_weight,
            2,
        )

    if not include_explanations:
        reasons = []
    elif not reasons:
        reasons.append("Recommended based on current catalogue fit and profile signals.")

    return RecommendedProduct(
        product_id=product.product_id,
        name=product.name,
        brand=product.brand or "Brand not listed",
        category=product.category,
        price=product.price,
        rating=product.rating,
        image_url=product.image_url,
        score=_clamp(final_score),
        confidence=_clamp(final_confidence),
        reasons=list(dict.fromkeys(reasons)),
        attribute_breakdown={key: round(_clamp(value), 2) for key, value in attribute_breakdown.items()},
        score_breakdown={key: round(value, 2) for key, value in score_breakdown.items()},
        catalogue_source=product.catalogue_source,
        availability_status=product.availability_status,
        price_is_historical=product.price_is_historical,
        rating_count=product.rating_count,
    )


def _battery_max_hours_by_category(
    products: Sequence[RecommendationCandidate],
) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for product in products:
        attributes = product.attributes if isinstance(product.attributes, dict) else {}
        value = attributes.get("battery_life_hours")
        if value is None:
            continue
        try:
            hours = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(hours) or hours <= 0:
            continue
        category = product.category.strip().casefold()
        maxima[category] = max(maxima.get(category, 0.0), hours)
    return maxima


def _priority_reason_dimensions(
    preference_overrides: Mapping[PreferenceOverrideDimension, int],
) -> list[str]:
    dimension_indexes = {dimension: index for index, dimension in enumerate(DIMENSION_ORDER)}
    return sorted(
        (
            dimension
            for dimension, priority in preference_overrides.items()
            if priority >= 60
        ),
        key=lambda dimension: (
            -preference_overrides[dimension],
            dimension_indexes[dimension],
        ),
    )


def generate_recommendations(
    db: Session,
    request: RecommendationRequest,
    *,
    preference_overrides: Mapping[PreferenceOverrideDimension, int] | None = None,
    required_connectivity: ConnectivityRequirement | None = None,
    required_product_type: ProductTypeRequirement | None = None,
    min_battery_life_hours: float | None = None,
    require_battery_life_data: bool = False,
    prioritize_children_headphones: bool = False,
    excluded_product_ids: Collection[str] = (),
    product_snapshot: Sequence[
        Product | AmazonCatalogueProduct | RecommendationCandidate
    ]
    | None = None,
) -> RecommendationResponse:
    """Generate deterministic, explainable product recommendations for a user."""
    overrides = dict(preference_overrides or {})
    unsupported_dimensions = set(overrides) - {
        "comfort_priority",
        "brand_loyalty",
        "battery_life_priority",
        "portability_preference",
    }
    if unsupported_dimensions:
        raise ValueError(f"Unsupported recommendation preference overrides: {sorted(unsupported_dimensions)}")

    profile = db.get(UserProfile, request.user_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User profile {request.user_id} not found",
        )

    dna_query = select(ShopperDNA).where(ShopperDNA.user_id == profile.id)
    dna = db.scalar(dna_query)
    if dna is None:
        dna = ShopperDNA(
            user_id=profile.id,
            inferred_preferences={dimension: 50 for dimension in DIMENSION_ORDER},
            confidence_scores={dimension: 0 for dimension in DIMENSION_ORDER},
            preference_sources={dimension: "default" for dimension in DIMENSION_ORDER},
        )

    effective_max_budget = (
        request.max_budget_override if request.max_budget_override is not None else profile.max_budget
    )
    effective_min_budget = max(
        (
            value
            for value in (profile.min_budget, request.min_budget_override)
            if value is not None
        ),
        default=None,
    )
    requested_category = request.effective_category

    if product_snapshot is None:
        curated_products = db.scalars(select(Product)).all()
        amazon_products = db.scalars(select(AmazonCatalogueProduct)).all()
        all_products = [
            *(_as_candidate(product) for product in curated_products),
            *(_as_candidate(product) for product in amazon_products),
        ]
    else:
        all_products = [_as_candidate(product) for product in product_snapshot]
    candidate_count = len(all_products)

    battery_products = (
        [
            product
            for product in all_products
            if _candidate_product_type(product) == required_product_type
        ]
        if required_product_type is not None
        else all_products
    )
    battery_maxima = _battery_max_hours_by_category(battery_products)

    ranked_products: list[RecommendedProduct] = []
    excluded: list[ExcludedProduct] = []
    previously_shown = load_recommendation_products(
        db,
        list(excluded_product_ids),
    )
    seen_identities = {
        identity
        for product in previously_shown
        if (identity := _candidate_identity(product)) is not None
    }
    excluded_ids = {
        _canonical_recommendation_id(product_id)
        for product_id in excluded_product_ids
    }

    for product in all_products:
        exclusion = _compute_exclusion_reason(
            product,
            request=request,
            profile=profile,
            require_available=request.require_available,
            required_connectivity=required_connectivity,
            required_product_type=required_product_type,
            min_battery_life_hours=min_battery_life_hours,
            require_battery_life_data=require_battery_life_data,
        )
        if exclusion is not None:
            excluded.append(exclusion)
            continue
        if _canonical_recommendation_id(product.product_id) in excluded_ids:
            excluded.append(
                ExcludedProduct(
                    product_id=product.product_id,
                    name=product.name,
                    reason_code="already_recommended",
                    message="Product was already shown in this conversation.",
                    catalogue_source=product.catalogue_source,
                )
            )
            continue

        identity = _candidate_identity(product)
        if identity is not None and identity in seen_identities:
            excluded.append(
                ExcludedProduct(
                    product_id=product.product_id,
                    name=product.name,
                    reason_code="duplicate_product",
                    message="A matching product identity was already considered.",
                    catalogue_source=product.catalogue_source,
                )
            )
            continue
        if identity is not None:
            seen_identities.add(identity)

        ranked_products.append(
            _score_product(
                product,
                profile=profile,
                dna=dna,
                max_budget=effective_max_budget,
                min_budget=effective_min_budget,
                min_rating=request.min_rating,
                requested_category=requested_category,
                required_connectivity=required_connectivity,
                required_product_type=required_product_type,
                battery_max_hours=battery_maxima.get(product.category.strip().casefold()),
                include_explanations=request.include_explanations,
                preference_overrides=overrides,
            )
        )

    is_headphone_request = bool(
        requested_category
        and requested_category.strip().casefold()
        in {"headphones", "headphones & earbuds"}
    )
    ranked_products.sort(
        key=lambda item: (
            bool(
                is_headphone_request
                and not prioritize_children_headphones
                and _CHILDREN_HEADPHONE_TITLE_PATTERN.search(item.name)
            ),
            -item.score,
            item.product_id,
        )
    )
    selected = ranked_products[: request.limit]

    explanation_summary = (
        f"Returned {len(selected)} recommendation(s) for user {request.user_id} "
        f"after evaluating {candidate_count} candidate product(s)."
    )

    return RecommendationResponse(
        user_id=request.user_id,
        generated_at=datetime.now(timezone.utc),
        requested_limit=request.limit,
        products=selected,
        excluded_products=excluded,
        explanation_summary=explanation_summary,
        has_more=len(ranked_products) > request.limit,
    )
