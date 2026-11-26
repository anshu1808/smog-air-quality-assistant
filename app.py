import streamlit as st

from models import HouseholdProfile
from graph_workflow import smog_graph

st.set_page_config(
    page_title="Winter Smog Assistant",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Winter Smog Assistant")
st.markdown(
    "Get personalized recommendations based on live AQI, your commute habits, and household profile."
)

# ==========================
# User Inputs
# ==========================

st.subheader("📍 Household Profile")

col1, col2 = st.columns(2)

with col1:

    city = st.text_input(
        "City",
        value="Delhi"
    )

    latitude = st.number_input(
        "Latitude",
        value=28.61,
        format="%.4f"
    )

    longitude = st.number_input(
        "Longitude",
        value=77.20,
        format="%.4f"
    )

with col2:

    commute_mode = st.selectbox(
        "Commute Mode",
        [
            "car",
            "two_wheeler",
            "transit",
            "walk_cycle",
            "remote"
        ]
    )

    daily_commute_km = st.slider(
        "Daily Commute Distance (KM)",
        min_value=0,
        max_value=100,
        value=15
    )

    has_sensitive_members = st.checkbox(
        "Sensitive household members (children, elderly, asthma patients)"
    )

    has_air_purifier = st.checkbox(
        "Air purifier available"
    )

# ==========================
# Generate
# ==========================

if st.button("🚀 Generate Advisory"):

    profile = HouseholdProfile(
        city=city,
        latitude=latitude,
        longitude=longitude,
        commute_mode=commute_mode,
        daily_commute_km=daily_commute_km,
        has_sensitive_members=has_sensitive_members,
        has_air_purifier=has_air_purifier
    )

    with st.spinner("Fetching AQI and generating recommendations..."):

        result = smog_graph.invoke(
            {
                "profile": profile,
                "aqi_payload": {},
                "computed_metrics": {}
            },
            config={
                "configurable": {
                    "thread_id": "streamlit_user"
                }
            }
        )

    advisory = result["advisory"]

    # ==========================
    # AQI Dashboard
    # ==========================

    st.subheader("🌍 Air Quality Snapshot")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "US AQI",
        advisory.current_aqi_us
    )

    c2.metric(
        "PM2.5 (µg/m³)",
        advisory.pm25_ug_m3
    )

    c3.metric(
        "Severity",
        advisory.severity_band
    )

    # ==========================
    # Summary
    # ==========================

    st.subheader("📋 Summary")

    st.info(advisory.summary)

    # ==========================
    # Impact
    # ==========================

    st.subheader("🌱 Impact of Your Choices")

    impact = advisory.impact_summary

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Daily CO₂ Saved (kg)",
        impact.daily_co2_saved_kg
    )

    col2.metric(
        "Annual CO₂ Saved (kg)",
        impact.annual_co2_saved_kg
    )

    col3.metric(
        "Indoor PM2.5",
        impact.estimated_indoor_pm25
    )

    # ==========================
    # Recommendations
    # ==========================

    st.subheader("✅ Recommended Actions")

    for rec in advisory.recommendations:

        with st.expander(rec.title):

            st.write(
                f"**Category:** {rec.category}"
            )

            st.write(
                f"**Impact:** {rec.estimated_impact}"
            )

            st.write("**Steps:**")

            for step in rec.action_steps:
                st.write(f"• {step}")

            st.caption(
                f"Tradeoff: {rec.tradeoff_or_note}"
            )

    # ==========================
    # Assumptions
    # ==========================

    st.subheader("📝 Assumptions")

    for item in advisory.assumptions:
        st.write(f"• {item}")

    # ==========================
    # Disclaimer
    # ==========================

    st.warning(advisory.disclaimer)