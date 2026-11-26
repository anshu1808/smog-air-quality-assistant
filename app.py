import streamlit as st

from models import HouseholdProfile

from graph_workflow import smog_graph
st.title(
    "🌫️ Winter Smog Assistant")

city = st.text_input(
    "City",
    "Delhi"
)

latitude = st.number_input(
    "Latitude",
    value=28.61
)

longitude = st.number_input(
    "Longitude",
    value=77.20
)

if st.button(
    "Generate Advisory"
):
    profile = HouseholdProfile(
        city=city,
        latitude=latitude,
        longitude=longitude
    )

    result = smog_graph.invoke({
        "profile": profile,
        "aqi_payload": {},
        "computed_metrics": {},
    },
    config = {
        "configurable": {"thread_id": "streamlit_user"}
    }
    )

    advisory = result["advisory"]
    st.write(advisory.summery)
    st.subheader("Smog Advisory Response")
    st.json(advisory.model_dump())