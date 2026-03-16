from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import requests


WORLD_BANK_API_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
WORLD_BANK_SOURCE_URL = "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation"
COUNTRY_CODE = "RWA"
INDICATORS = {
    "population": "SP.POP.TOTL",
    "fertility_rate": "SP.DYN.TFRT.IN",
    "life_expectancy": "SP.DYN.LE00.MA.IN",
    "net_migration": "SM.POP.NETM",
}


@dataclass(frozen=True)
class PopulationModel:
    min_year: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    holdout_mae: float
    holdout_mape: float
    baseline_holdout_mae: float
    baseline_holdout_mape: float
    latest_actual_population: int
    latest_actual_year: int
    feature_names: tuple[str, ...]
    feature_importance: tuple[float, ...]


def _to_float(value: str | int | float | None) -> float | None:
    """Convert World Bank string values into floats when possible."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned == "..":
        return None
    return float(cleaned)


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _fetch_indicator_series(
    indicator_code: str,
    country_code: str = COUNTRY_CODE,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict[int, float]:
    """Fetch one indicator series from the World Bank API."""
    client = session or requests.Session()
    url = WORLD_BANK_API_BASE.format(country=country_code, indicator=indicator_code)
    response = client.get(
        url,
        params={"format": "json", "per_page": 20000, "source": 2},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError(f"Unexpected World Bank API response for indicator {indicator_code}.")

    values: dict[int, float] = {}
    for item in payload[1]:
        year = _to_float(item.get("date"))
        value = _to_float(item.get("value"))
        if year is not None and value is not None:
            values[int(year)] = value

    if not values:
        raise ValueError(f"No usable data returned for indicator {indicator_code}.")

    return values


def load_rwanda_population_data_live() -> list[dict[str, float]]:
    """Load overlapping Rwanda indicator records needed for model training."""
    with requests.Session() as session:
        population = _fetch_indicator_series(INDICATORS["population"], session=session)
        fertility = _fetch_indicator_series(INDICATORS["fertility_rate"], session=session)
        life_expectancy = _fetch_indicator_series(INDICATORS["life_expectancy"], session=session)
        migration = _fetch_indicator_series(INDICATORS["net_migration"], session=session)

    common_years = sorted(set(population) & set(fertility) & set(life_expectancy) & set(migration))
    if not common_years:
        raise ValueError("No overlapping Rwanda records were returned by the World Bank API.")

    records: list[dict[str, float]] = []
    for year in common_years:
        records.append(
            {
                "year": year,
                "fertility_rate": fertility[year],
                "life_expectancy": life_expectancy[year],
                "net_migration": migration[year],
                "population": population[year],
            }
        )
    return records


def _build_features(
    years: Iterable[float],
    fertility: Iterable[float],
    life_expectancy: Iterable[float],
    migration: Iterable[float],
    previous_population: Iterable[float],
    min_year: int,
) -> np.ndarray:
    """Build the feature matrix used by the regression model."""
    years_array = np.asarray(list(years), dtype=float)
    fertility_array = np.asarray(list(fertility), dtype=float)
    life_expectancy_array = np.asarray(list(life_expectancy), dtype=float)
    migration_array = np.asarray(list(migration), dtype=float)
    previous_population_array = np.asarray(list(previous_population), dtype=float)

    year_offset = years_array - float(min_year)
    signed_migration = np.vectorize(_signed_log1p)(migration_array)

    return np.column_stack(
        [
            year_offset,
            fertility_array,
            life_expectancy_array,
            signed_migration,
            previous_population_array / 10_000_000.0,
            fertility_array * life_expectancy_array,
        ]
    )


FEATURE_NAMES = (
    "Year offset",
    "Fertility rate",
    "Life expectancy",
    "Net migration",
    "Previous population",
    "Fertility x life expectancy",
)


def _fit_linear_model(features: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a normalized ridge-style linear model."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std

    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0

    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return mean, std, weights


def _predict_from_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = (features - mean) / std
    design = np.column_stack([np.ones(len(normalized)), normalized])
    return design @ weights


def train_population_model(records: list[dict[str, float]]) -> PopulationModel:
    """Train a population growth model from historical indicator records."""
    years = np.array([row["year"] for row in records], dtype=float)
    fertility = np.array([row["fertility_rate"] for row in records], dtype=float)
    life_expectancy = np.array([row["life_expectancy"] for row in records], dtype=float)
    migration = np.array([row["net_migration"] for row in records], dtype=float)
    population = np.array([row["population"] for row in records], dtype=float)

    min_year = int(years.min())
    growth_years = years[1:]
    growth_fertility = fertility[1:]
    growth_life_expectancy = life_expectancy[1:]
    growth_migration = migration[1:]
    previous_population = population[:-1]
    next_population = population[1:]
    growth_rate = np.log(next_population / previous_population)
    features = _build_features(
        growth_years,
        growth_fertility,
        growth_life_expectancy,
        growth_migration,
        previous_population,
        min_year,
    )

    split_index = max(8, len(features) // 5)
    train_features = features[:-split_index]
    test_features = features[-split_index:]
    train_targets = growth_rate[:-split_index]
    test_population = next_population[-split_index:]

    train_mean, train_std, train_weights = _fit_linear_model(train_features, train_targets)
    holdout_predictions = []
    baseline_holdout_predictions = []
    current_population = previous_population[-split_index]
    baseline_growth = float(train_targets[-1])
    for feature_row in test_features:
        adjusted_feature_row = feature_row.copy()
        adjusted_feature_row[4] = current_population / 10_000_000.0
        growth_prediction = _predict_from_features(
            adjusted_feature_row.reshape(1, -1),
            train_mean,
            train_std,
            train_weights,
        )[0]
        current_population = max(float(current_population * math.exp(growth_prediction)), 0.0)
        holdout_predictions.append(current_population)
        baseline_population = max(
            float(baseline_holdout_predictions[-1] if baseline_holdout_predictions else previous_population[-split_index])
            * math.exp(baseline_growth),
            0.0,
        )
        baseline_holdout_predictions.append(baseline_population)

    holdout_predictions_array = np.array(holdout_predictions, dtype=float)
    baseline_holdout_predictions_array = np.array(baseline_holdout_predictions, dtype=float)

    holdout_mae = float(np.mean(np.abs(holdout_predictions_array - test_population)))
    holdout_mape = float(
        np.mean(np.abs((holdout_predictions_array - test_population) / np.maximum(test_population, 1.0))) * 100
    )
    baseline_holdout_mae = float(np.mean(np.abs(baseline_holdout_predictions_array - test_population)))
    baseline_holdout_mape = float(
        np.mean(
            np.abs(
                (baseline_holdout_predictions_array - test_population) / np.maximum(test_population, 1.0)
            )
        )
        * 100
    )

    feature_mean, feature_std, weights = _fit_linear_model(features, growth_rate)
    importance_weights = np.abs(weights[1:])
    importance_total = float(importance_weights.sum())
    if importance_total == 0:
        feature_importance = tuple(0.0 for _ in FEATURE_NAMES)
    else:
        feature_importance = tuple(float(value / importance_total * 100) for value in importance_weights)

    return PopulationModel(
        min_year=min_year,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=weights,
        holdout_mae=holdout_mae,
        holdout_mape=holdout_mape,
        baseline_holdout_mae=baseline_holdout_mae,
        baseline_holdout_mape=baseline_holdout_mape,
        latest_actual_population=int(population[-1]),
        latest_actual_year=int(years[-1]),
        feature_names=FEATURE_NAMES,
        feature_importance=feature_importance,
    )


def build_default_model() -> tuple[PopulationModel, list[dict[str, float]]]:
    """Build the live model and return it with the historical source records."""
    records = load_rwanda_population_data_live()
    model = train_population_model(records)
    return model, records


def _forecast_population_path(
    model: PopulationModel,
    start_year: int,
    target_year: int,
    fertility: float,
    life_exp: float,
    migration: float,
    starting_population: float,
) -> list[tuple[int, int]]:
    """Generate yearly population forecasts from the latest actual population."""
    projected_points: list[tuple[int, int]] = []
    current_population = float(starting_population)

    for year in range(start_year + 1, target_year + 1):
        features = _build_features(
            [year],
            [fertility],
            [life_exp],
            [migration],
            [current_population],
            model.min_year,
        )
        growth_prediction = _predict_from_features(features, model.feature_mean, model.feature_std, model.weights)[0]
        current_population = max(float(current_population * math.exp(growth_prediction)), 0.0)
        projected_points.append((year, int(round(current_population))))

    return projected_points


def predict_population(
    model: PopulationModel,
    year: float,
    fertility: float,
    life_exp: float,
    migration: float,
) -> int:
    """Predict Rwanda's population for a single future year."""
    forecast_year = int(year)
    if forecast_year <= model.latest_actual_year:
        return model.latest_actual_population

    forecast_path = _forecast_population_path(
        model=model,
        start_year=model.latest_actual_year,
        target_year=forecast_year,
        fertility=fertility,
        life_exp=life_exp,
        migration=migration,
        starting_population=float(model.latest_actual_population),
    )
    return forecast_path[-1][1]


def build_forecast_series(
    model: PopulationModel,
    history: list[dict[str, float]],
    target_year: int,
    fertility: float,
    life_exp: float,
    migration: float,
) -> list[dict[str, int | str]]:
    """Build a chart-ready historical plus forecast series."""
    latest_year = int(history[-1]["year"])
    series = [
        {"Year": int(row["year"]), "Population": int(row["population"]), "Series": "Historical"}
        for row in history
    ]

    forecast_path = _forecast_population_path(
        model=model,
        start_year=latest_year,
        target_year=target_year,
        fertility=fertility,
        life_exp=life_exp,
        migration=migration,
        starting_population=float(model.latest_actual_population),
    )
    for year, population in forecast_path:
        series.append(
            {
                "Year": year,
                "Population": population,
                "Series": "Forecast",
            }
        )

    return series


def build_scenario_comparison_series(
    model: PopulationModel,
    history: list[dict[str, float]],
    target_year: int,
    fertility: float,
    life_exp: float,
    migration: float,
) -> list[dict[str, int | str]]:
    """Build low, selected, and high-growth forecast paths for comparison."""
    latest = history[-1]
    scenarios = {
        "Selected scenario": (fertility, life_exp, migration),
        "Lower-growth scenario": (
            max(1.0, fertility - 0.2),
            max(45.0, life_exp - 2.0),
            migration - 5_000,
        ),
        "Higher-growth scenario": (
            min(8.5, fertility + 0.2),
            min(90.0, life_exp + 2.0),
            migration + 5_000,
        ),
        "Recent-trend scenario": (
            float(latest["fertility_rate"]),
            float(latest["life_expectancy"]),
            float(latest["net_migration"]),
        ),
    }

    comparison_series = [
        {"Year": int(row["year"]), "Population": int(row["population"]), "Series": "Historical"}
        for row in history
    ]
    latest_year = int(latest["year"])

    for label, (scenario_fertility, scenario_life_exp, scenario_migration) in scenarios.items():
        forecast_path = _forecast_population_path(
            model=model,
            start_year=latest_year,
            target_year=target_year,
            fertility=float(scenario_fertility),
            life_exp=float(scenario_life_exp),
            migration=float(scenario_migration),
            starting_population=float(model.latest_actual_population),
        )
        for year, population in forecast_path:
            comparison_series.append(
                {
                    "Year": year,
                    "Population": population,
                    "Series": label,
                }
            )

    return comparison_series


def build_feature_importance_rows(model: PopulationModel) -> list[dict[str, float | str]]:
    """Return feature importance values in display-ready order."""
    importance_rows = [
        {"Feature": feature, "Importance": round(importance, 2)}
        for feature, importance in zip(model.feature_names, model.feature_importance)
    ]
    return sorted(importance_rows, key=lambda row: float(row["Importance"]), reverse=True)
