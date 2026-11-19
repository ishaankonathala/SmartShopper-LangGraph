"""CLI entrypoint: python -m app.data.seed_cli

Seeds the demo catalogue and demo shopper profile into the configured SQLite database.
Safe to run multiple times — does not duplicate SKUs or touch user data.
"""

from app.database.session import SessionLocal
from app.services.seed import seed_demo_profile, seed_products


def main() -> None:
    db = SessionLocal()
    try:
        product_result = seed_products(db)
        demo_profile = seed_demo_profile(db)
        print("Database seed complete (demo catalogue and profile).")
        print(
            f"  catalogue: inserted={product_result['inserted']} "
            f"updated={product_result['updated']} "
            f"skipped={product_result['skipped']} "
            f"seed_file={product_result['total_in_seed_file']}"
        )
        print(
            f"  demo_shopper: id={demo_profile.id} "
            f"name={demo_profile.name} "
            f"max_budget=₹{demo_profile.max_budget:,.0f}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
