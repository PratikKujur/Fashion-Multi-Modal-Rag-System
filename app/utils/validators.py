from typing import Optional

def validate_category(category: Optional[str]) -> bool:
    valid_categories = ["shirt", "pants", "dress", "shoes", "jacket", "accessories"]
    return category is None or category.lower() in valid_categories

def validate_limit(limit: int, max_limit: int = 50) -> int:
    return min(max(1, limit), max_limit)
