"""Idempotent catalogue and demo profile seeding."""

from typing import Final
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.dna_defaults import (
    DEFAULT_CONFIDENCE_SCORES,
    DEFAULT_INFERRED_PREFERENCES,
    DEFAULT_PREFERENCE_SOURCES,
)
from app.data.catalogue import DEMO_PRODUCTS
from app.models.product import Product
from app.models.shopper_dna import ShopperDNA
from app.models.user_profile import UserProfile

DEMO_USER_ID: Final[int] = 1


def seed_products(db: Session) -> dict[str, int]:
    """Insert missing demo products and backfill absent catalogue attributes.

    Existing attribute values and all other product fields are preserved.
    Returns counts: inserted, updated, skipped, total_in_seed_file.
    """
    inserted = 0
    updated = 0
    skipped = 0

    for item in DEMO_PRODUCTS:
        existing = db.scalar(select(Product).where(Product.product_id == item["product_id"]))
        if existing is not None:
            if not isinstance(existing.attributes, dict):
                raise ValueError(
                    f"Existing product {existing.product_id} has invalid catalogue attributes."
                )

            missing_attributes = {
                key: value
                for key, value in item["attributes"].items()
                if key not in existing.attributes
            }
            if missing_attributes:
                existing.attributes = {
                    **existing.attributes,
                    **missing_attributes,
                }
                updated += 1
            else:
                skipped += 1
            continue

        db.add(
            Product(
                product_id=item["product_id"],
                name=item["name"],
                brand=item["brand"],
                category=item["category"],
                price=item["price"],
                rating=item["rating"],
                description=item["description"],
                attributes=dict(item["attributes"]),
                stock_quantity=item["stock_quantity"],
                is_active=item["is_active"],
                image_url=item["image_url"],
            )
        )
        inserted += 1

    if inserted or updated:
        db.commit()

    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "total_in_seed_file": len(DEMO_PRODUCTS),
    }


def seed_demo_profile(db: Session) -> UserProfile:
    """Ensure the default demo shopper profile exists with ID=1.

    Safe and idempotent — preserves existing profile data without overwriting.
    """
    existing = db.get(UserProfile, DEMO_USER_ID)
    if existing is not None:
        return existing

    email = "demo@smartshopper.local"
    email_conflict = db.scalar(select(UserProfile).where(UserProfile.email == email))
    if email_conflict is not None:
        email = None

    profile = UserProfile(
        id=DEMO_USER_ID,
        name="Demo Shopper",
        email=email,
        max_budget=6500.0,
        min_budget=1000.0,
        preferred_categories=["Headphones"],
        preferred_brands=["Sony"],
        disliked_attributes=["bulky"],
    )
    profile.dna = ShopperDNA(
        inferred_preferences=dict(DEFAULT_INFERRED_PREFERENCES),
        confidence_scores=dict(DEFAULT_CONFIDENCE_SCORES),
        preference_sources=dict(DEFAULT_PREFERENCE_SOURCES),
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile
