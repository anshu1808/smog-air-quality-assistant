from typing import Literal, TypedDict
from pydantic import BaseModel

class HouseholdProfile(BaseModel):
    city: str
    latitude: float
    longitude: float

    commute_mode: Literal[
        "car",
        "two_wheeler",
        "transit",
        "walk_cycle",
        "remote"
    ] = "two_wheeler"

    daily_commute_km: float = 15.0

    has_sensitive_members: bool = False

    has_air_purifier: bool = False

class ActionItem(BaseModel):
    title: str

    category: Literal[
        "daily_routine",
        "indoor_air_hack",
        "sustainable_alternative"
    ]

    action_steps: list[str]

    estimated_impact: str

    tradeoff_or_note: str

class ImpactSummary(BaseModel):

    daily_co2_saved_kg: float

    annual_co2_saved_kg: float

    outdoor_pm25: float

    estimated_indoor_pm25: float

    exposure_reduction_pct: float

class SmogAdvisoryResponse(BaseModel):

    city: str
    current_aqi_us: float | None
    pm25_ug_m3: float | None
    severity_band: str
    summary: str

    impact_summary: ImpactSummary

    recommendations: list[ActionItem]

    assumptions: list[str]

    disclaimer: str

class GraphState(TypedDict):
    profile: HouseholdProfile
    aqi_payload: dict
    computed_metrics: dict
    advisory: SmogAdvisoryResponse

