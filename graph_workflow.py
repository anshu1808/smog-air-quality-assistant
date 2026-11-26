import httpx

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import (
    InMemorySaver
)

from models import (
    HouseholdProfile,
    SmogAdvisoryResponse,
    GraphState
)

from config import llm

def fetch_aqi_node(state):
    print("STATE received:", state)
    profile = state["profile"]

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": profile.latitude,
        "longitude": profile.longitude,
        "current": "pm10,pm2_5,nitrogen_dioxide,us_aqi",
    }

    data = httpx.get(
        url,
        params=params
    ).json()

    current = data["current"]

    return {
        "aqi_payload": {
            "pm25": current["pm2_5"],
            "us_aqi": current["us_aqi"]
        }
    }

def classify_severity(pm25):

    if pm25 >= 150:
        return "severe"

    if pm25 >= 55.5:
        return "unhealthy"

    if pm25 >= 35.5:
        return "unhealthy_sensitive"

    if pm25 >= 12:
        return "moderate"

    return "low"

def compute_metrics(profile, pm25):

    emission_factors = {
        "car": 0.17,
        "two_wheeler": 0.06,
        "transit": 0.04,
        "walk_cycle": 0.0,
        "remote": 0.0
    }

    ef = emission_factors[profile.commute_mode]

    avoidable_km = round(profile.daily_commute_km * 0.30, 2)

    daily_co2_saved = round(avoidable_km * ef, 3)

    annual_co2_saved = round(daily_co2_saved * 365, 2)

    indoor_reduction_pct = (
        75 if profile.has_air_purifier
        else 45
    )

    estimated_indoor_pm25 = round(
        pm25 * (1 - indoor_reduction_pct / 100),
        2
    )

    return {
        "avoidable_km": avoidable_km,

        "daily_co2_saved_kg": daily_co2_saved,

        "annual_co2_saved_kg": annual_co2_saved,

        "outdoor_pm25": pm25,

        "estimated_indoor_pm25":
            estimated_indoor_pm25,

        "indoor_exposure_reduction_pct":
            indoor_reduction_pct,

        "severity":
            classify_severity(pm25)
    }

def compute_impact_node(state):

    profile = state["profile"]

    aqi = state["aqi_payload"]

    metrics = compute_metrics(
        profile,
        aqi["pm25"]
    )

    return {
        "computed_metrics": metrics
    }

def generate_advisory_node(state):

    profile = state["profile"]

    metrics = state["computed_metrics"]

    structured_llm = llm.with_structured_output(
        SmogAdvisoryResponse
    )

    response = structured_llm.invoke(
        f"""
        Household Profile:
        {profile}

        AQI:
        {state["aqi_payload"]}


        Metrics:
        {metrics}

        Special Alert:
        {metrics.get("special_alert")}

        Create personalized smog guidance.

        Explain:

        1. How much CO2 the user saves daily.

        2. How much CO2 the user saves annually.

        3. Estimated indoor PM2.5 reduction.

        4. Difference between exposure reduction
        and pollution reduction.

        5. Give 3-5 actionable recommendations.
        """
    )
    return {
        "advisory": response
    }

def route_by_severity(state):

    severity = state["computed_metrics"]["severity"]

    print("Severity:", severity)

    if severity in [
        "unhealthy",
        "severe",
        "unhealthy_sensitive"
    ]:
        return "high_risk"

    return "normal_risk"

def high_risk_node(state):

    return {
        "computed_metrics": {
            **state["computed_metrics"],
            "risk_level": "high",
            "special_alert":
                "Avoid outdoor exposure whenever possible."
        }
    }

def normal_risk_node(state):

    return {
        "computed_metrics": {
            **state["computed_metrics"],
            "risk_level": "normal",
            "special_alert":
                "Outdoor activity allowed with precautions."
        }
    }

memory = InMemorySaver()

workflow = StateGraph(GraphState)

# nodes
# edges
# conditional routing

workflow.add_node(
    "fetch_aqi",
    fetch_aqi_node
)

workflow.add_node(
    "compute_impact",
    compute_impact_node
)

workflow.add_node(
    "generate_advisory",
    generate_advisory_node
)

workflow.add_node(
    "high_risk",
    high_risk_node
)

workflow.add_node(
    "normal_risk",
    normal_risk_node
)

workflow.add_edge(
    START,
    "fetch_aqi"
)

workflow.add_edge(
    "fetch_aqi",
    "compute_impact"
)
"""
workflow.add_edge(
    "compute_impact",
    "generate_advisory"
)
"""
workflow.add_conditional_edges(
    "compute_impact",
    route_by_severity,
    {
        "high_risk": "high_risk",
        "normal_risk": "normal_risk"
    }
)

workflow.add_edge(
    "high_risk",
    "generate_advisory"
)

workflow.add_edge(
    "normal_risk",
    "generate_advisory"
)

workflow.add_edge(
    "generate_advisory",
    END
)


smog_graph = workflow.compile(
    checkpointer=memory
)