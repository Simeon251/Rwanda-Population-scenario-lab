from __future__ import annotations

import json
from pathlib import Path

from utils import WORLD_BANK_SOURCE_URL, build_default_model, build_forecast_series


def main() -> None:
    model, history = build_default_model()
    latest = history[-1]
    target_year = latest["year"] + 6
    forecast = build_forecast_series(
        model,
        history,
        int(target_year),
        float(latest["fertility_rate"]),
        float(latest["life_expectancy"]),
        float(latest["net_migration"]),
    )

    summary = {
        "source": WORLD_BANK_SOURCE_URL,
        "latest_actual_year": model.latest_actual_year,
        "latest_actual_population": model.latest_actual_population,
        "holdout_mae": round(model.holdout_mae, 2),
        "holdout_mape": round(model.holdout_mape, 2),
        "baseline_holdout_mae": round(model.baseline_holdout_mae, 2),
        "baseline_holdout_mape": round(model.baseline_holdout_mape, 2),
        "feature_importance": [
            {"feature": feature, "importance": round(importance, 2)}
            for feature, importance in zip(model.feature_names, model.feature_importance)
        ],
        "sample_forecast": forecast[-1],
    }

    Path("model").mkdir(exist_ok=True)
    Path("model/model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
