from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import (
    WORLD_BANK_SOURCE_URL,
    build_default_model,
    build_feature_importance_rows,
    build_forecast_series,
    build_scenario_comparison_series,
    predict_population,
)


st.set_page_config(
    page_title="Rwanda Population Scenario Lab",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_app_state():
    return build_default_model()


st.title("Rwanda Population Scenario Lab")
st.write(
    "Explore how fertility, life expectancy, and net migration can influence Rwanda's population "
    "trajectory using live World Bank indicator data."
)

refresh_col, source_col = st.columns([0.2, 0.8])
with refresh_col:
    if st.button("Refresh data"):
        load_app_state.clear()
with source_col:
    st.caption(f"Source: [World Bank Indicators API]({WORLD_BANK_SOURCE_URL})")

try:
    model, history = load_app_state()
except Exception as exc:
    st.error(
        "Unable to load live Rwanda data from the World Bank API right now. "
        f"Details: {exc}"
    )
    st.stop()


latest = history[-1]
latest_year = int(latest["year"])
latest_population = int(latest["population"])

st.caption(
    f"Historical data coverage: {history[0]['year']} to {latest_year}. "
    f"Latest World Bank population value: {latest_population:,}."
)

with st.sidebar:
    st.header("Scenario Controls")
    year = st.slider("Target year", latest_year + 1, 2100, min(latest_year + 6, 2100))
    fertility = st.slider(
        "Fertility rate",
        min_value=1.0,
        max_value=8.5,
        value=float(latest["fertility_rate"]),
        step=0.01,
        help="Average number of births per woman from the World Bank fertility indicator.",
    )
    life_exp = st.slider(
        "Life expectancy",
        min_value=45.0,
        max_value=90.0,
        value=float(latest["life_expectancy"]),
        step=0.1,
        help="Life expectancy at birth in years.",
    )
    migration = st.number_input(
        "Net migration",
        value=int(latest["net_migration"]),
        step=500,
        help="Positive means more people move in than out. Negative values are supported.",
    )

prediction = predict_population(model, year, fertility, life_exp, migration)
delta = prediction - latest_population
growth_pct = (delta / latest_population) * 100

metric_col, insight_col = st.columns([1.1, 0.9])

with metric_col:
    st.subheader("Forecast")
    st.metric("Projected population", f"{prediction:,}", f"{delta:+,} vs {latest_year}")
    st.write(f"Projected change from {latest_year}: **{growth_pct:+.2f}%**")
    if model.holdout_mape < model.baseline_holdout_mape:
        st.caption(
            f"Model holdout MAPE: {model.holdout_mape:.2f}% "
            f"vs baseline {model.baseline_holdout_mape:.2f}%."
        )
    else:
        st.caption(
            f"Model holdout MAPE: {model.holdout_mape:.2f}% "
            f"(baseline: {model.baseline_holdout_mape:.2f}%)."
        )

with insight_col:
    st.subheader("Scenario Summary")
    st.write(f"Year: **{year}**")
    st.write(f"Fertility: **{fertility:.2f} births/woman**")
    st.write(f"Life expectancy: **{life_exp:.1f} years**")
    st.write(f"Net migration: **{migration:,}**")

if prediction > latest_population * 2:
    st.warning("This scenario more than doubles the latest recorded population, so treat it as an aggressive long-range estimate.")
elif prediction < latest_population * 0.8:
    st.info("This scenario projects slower growth than Rwanda's recent trend.")
else:
    st.success("This scenario stays within a moderate growth band relative to recent history.")

forecast_df = pd.DataFrame(build_forecast_series(model, history, year, fertility, life_exp, migration))
chart_df = forecast_df.pivot(index="Year", columns="Series", values="Population")

st.subheader("Historical Trend And Scenario Forecast")
st.line_chart(chart_df, height=380)

comparison_df = pd.DataFrame(
    build_scenario_comparison_series(model, history, year, fertility, life_exp, migration)
)
comparison_chart_df = comparison_df.pivot(index="Year", columns="Series", values="Population")

st.subheader("Scenario Comparison")
st.caption("These companion paths help you compare your chosen assumptions against lower-growth, higher-growth, and recent-trend cases.")
st.line_chart(comparison_chart_df, height=380)

with st.expander("Model notes and diagnostics"):
    st.write(
        "This version fetches Rwanda data live from the World Bank Indicators API and rebuilds the "
        "forecast model from the latest overlapping values on app load."
    )
    st.write(f"Holdout MAE: **{model.holdout_mae:,.0f} people**")
    st.write(f"Holdout MAPE: **{model.holdout_mape:.2f}%**")
    st.write(f"Baseline holdout MAE: **{model.baseline_holdout_mae:,.0f} people**")
    st.write(f"Baseline holdout MAPE: **{model.baseline_holdout_mape:.2f}%**")
    if model.holdout_mape <= model.baseline_holdout_mape:
        st.success("The scenario model currently outperforms the simple trend baseline on the holdout window.")
    else:
        st.warning(
            "The simple trend baseline currently beats the scenario model on the holdout window, "
            "so treat custom forecasts as exploratory scenario outputs rather than best-available predictions."
        )
    st.write("The cached API response refreshes every 12 hours, or immediately when you press Refresh data.")
    st.write("Use the output as a scenario estimate rather than an official census projection.")

st.subheader("Feature Importance")
feature_importance_df = pd.DataFrame(build_feature_importance_rows(model))
st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)

st.subheader("Recent Historical Data")
history_df = pd.DataFrame(history)
display_df = history_df.tail(8).copy()
display_df.columns = ["Year", "Fertility Rate", "Life Expectancy", "Net Migration", "Population"]
display_df["Population"] = display_df["Population"].astype(int)
display_df["Net Migration"] = display_df["Net Migration"].astype(int)
st.dataframe(display_df, use_container_width=True, hide_index=True)
