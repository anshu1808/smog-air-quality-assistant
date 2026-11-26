import os
from typing import Literal, TypedDict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# 1. API Request & Structured Output Schemas
# ---------------------------------------------------------------------------

class HouseholdProfile(BaseModel):
    city: str = Field(..., description="City name for context and display")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    commute_mode: Literal["car", "two_wheeler", "transit", "walk_cycle", "remote"] = "two_wheeler"
    daily_commute_km: float = Field(default=15.0, ge=0, le=300)
    has_sensitive_members: bool = Field(
        default=False,
        description="True if children, older adults, or people with respiratory conditions live in the home",
    )
    has_air_purifier: bool = False
    goal: str = Field(
        default="Reduce winter smog exposure and daily household emissions",
        description="Primary focus for the user or community group",
    )


class ActionItem(BaseModel):
    title: str
    category: Literal["daily_routine", "indoor_air_hack", "sustainable_alternative"]
    action_steps: list[str] = Field(..., min_length=1)
    estimated_impact: str = Field(..., description="Plain-language explanation of exposure or emission impact")
    tradeoff_or_note: str


class SmogAdvisoryResponse(BaseModel):
    city: str
    current_aqi_us: float | None
    pm25_ug_m3: float | None
    severity_band: Literal["low", "moderate", "unhealthy_sensitive", "unhealthy", "severe"]
    summary: str
    computed_metrics: dict[str, float | str]
    recommendations: list[ActionItem] = Field(..., min_length=3)
    assumptions: list[str]
    disclaimer: str


# ---------------------------------------------------------------------------
# 2. LangGraph State & Deterministic Helper Functions
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    profile: HouseholdProfile
    aqi_payload: dict
    computed_metrics: dict[str, float | str]
    advisory: SmogAdvisoryResponse


def classify_severity(pm25: float | None, us_aqi: float | None) -> str:
    val = pm25 if pm25 is not None else (us_aqi / 4.0 if us_aqi else 25.0)
    if val >= 150:
        return "severe"
    if val >= 55.5:
        return "unhealthy"
    if val >= 35.5:
        return "unhealthy_sensitive"
    if val >= 12.1:
        return "moderate"
    return "low"


def fetch_aqi_node(state: GraphState) -> dict:
    """Fetch current PM2.5, PM10, NO2, and US AQI from Open-Meteo Air Quality API."""
    profile = state["profile"]
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": profile.latitude,
        "longitude": profile.longitude,
        "current": "pm10,pm2_5,nitrogen_dioxide,us_aqi",
        "timezone": "auto",
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json().get("current", {})
            return {
                "aqi_payload": {
                    "source": "open-meteo",
                    "pm25": data.get("pm2_5"),
                    "pm10": data.get("pm10"),
                    "no2": data.get("nitrogen_dioxide"),
                    "us_aqi": data.get("us_aqi"),
                }
            }
    except Exception:
        # Fallback sample data for offline local testing
        return {
            "aqi_payload": {
                "source": "fallback_estimate",
                "pm25": 118.0,
                "pm10": 195.0,
                "no2": 46.0,
                "us_aqi": 182.0,
            }
        }


def compute_impact_node(state: GraphState) -> dict:
    """Compute transparent baseline figures in Python before LLM explanation."""
    profile = state["profile"]
    aqi = state["aqi_payload"]
    pm25 = float(aqi.get("pm25") or 60.0)
    us_aqi = float(aqi.get("us_aqi") or 120.0)

    # Simple illustrative emission factor (kg CO2 per km) for a 30% trip swap/consolidation
    emission_factors = {
        "car": 0.17,
        "two_wheeler": 0.06,
        "transit": 0.04,
        "walk_cycle": 0.0,
        "remote": 0.0,
    }
    ef = emission_factors.get(profile.commute_mode, 0.08)
    avoidable_km_per_day = round(profile.daily_commute_km * 0.30, 2)
    avoided_co2_kg_per_day = round(avoidable_km_per_day * ef, 3)

    # Illustrative indoor PM2.5 reduction factor when sealing leaks + filtration during smog
    indoor_reduction_pct = 75.0 if profile.has_air_purifier else 45.0
    estimated_indoor_pm25 = round(pm25 * (1.0 - indoor_reduction_pct / 100.0), 1)

    metrics = {
        "us_aqi": us_aqi,
        "outdoor_pm25_ug_m3": pm25,
        "severity_band": classify_severity(pm25, us_aqi),
        "avoidable_km_per_day_30pct_shift": avoidable_km_per_day,
        "estimated_co2_saved_kg_per_day": avoided_co2_kg_per_day,
        "estimated_indoor_pm25_ug_m3": estimated_indoor_pm25,
        "estimated_indoor_exposure_reduction_pct": indoor_reduction_pct,
    }
    return {"computed_metrics": metrics}


def generate_advisory_node(state: GraphState) -> dict:
    """Generate structured recommendations grounded in live AQI and computed metrics."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set in environment.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    structured_llm = llm.with_structured_output(SmogAdvisoryResponse)

    profile = state["profile"]
    aqi = state["aqi_payload"]
    metrics = state["computed_metrics"]

    system_prompt = (
        "You are a practical winter air-quality and low-emission assistant for families and communities. "
        "Use the provided AQI telemetry and pre-computed Python metrics. "
        "Provide concrete low-emission daily routines, indoor air quality hacks, and sustainable alternatives "
        "tailored to the user's location, commute mode, and household profile. "
        "Clearly distinguish personal exposure reduction from emission reduction, state key assumptions, "
        "and include a non-medical informational disclaimer."
    )

    user_prompt = (
        f"Household Profile: {profile.model_dump()}\n"
        f"AQI Telemetry: {aqi}\n"
        f"Pre-computed Metrics: {metrics}\n"
        "Generate a complete SmogAdvisoryResponse."
    )

    advisory = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    advisory.city = profile.city
    advisory.current_aqi_us = aqi.get("us_aqi")
    advisory.pm25_ug_m3 = aqi.get("pm25")
    advisory.computed_metrics = metrics
    return {"advisory": advisory}


# ---------------------------------------------------------------------------
# 3. Build LangGraph & FastAPI App
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)
workflow.add_node("fetch_aqi", fetch_aqi_node)
workflow.add_node("compute_impact", compute_impact_node)
workflow.add_node("generate_advisory", generate_advisory_node)

workflow.add_edge(START, "fetch_aqi")
workflow.add_edge("fetch_aqi", "compute_impact")
workflow.add_edge("compute_impact", "generate_advisory")
workflow.add_edge("generate_advisory", END)

smog_graph = workflow.compile()

app = FastAPI(title="Winter Smog & Low-Emission Assistant API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/advisory", response_model=SmogAdvisoryResponse)
def create_advisory(profile: HouseholdProfile) -> SmogAdvisoryResponse:
    try:
        result = smog_graph.invoke({
            "profile": profile,
            "aqi_payload": {},
            "computed_metrics": {},
        })
        return result["advisory"]
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Workflow execution failed: {exc}") from exc

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)