# src/api/routes_spatial.py
from fastapi import APIRouter

router = APIRouter(tags=["spatial"])

DISTRICTS = [
    {"district": "Auckland City", "category": "urban_core"},
    {"district": "North Shore", "category": "high_value"},
    {"district": "Manukau", "category": "urban"},
    {"district": "Waitakere", "category": "suburban"},
    {"district": "Rodney", "category": "outlying"},
    {"district": "Franklin", "category": "outlying"},
    {"district": "Papakura", "category": "outlying"},
]

@router.get("/spatial/districts")
def get_districts():
    return {
        "count": len(DISTRICTS),
        "districts": DISTRICTS,
    }
