# Rwanda Population Scenario Lab

Rwanda Population Scenario Lab is a Streamlit app for exploring how fertility, life expectancy, and net migration can influence Rwanda's future population. The app pulls live indicator data from the World Bank API, trains a lightweight forecasting model, and lets users test different demographic scenarios interactively.

## Features

- Live Rwanda demographic data from the World Bank Indicators API
- Interactive scenario controls for fertility, life expectancy, and net migration
- Forecast chart combining historical data with user-defined projections
- Simple model diagnostics with holdout MAE and MAPE
- Rebuildable model summary artifact in `model/model_summary.json`

## Project Structure

```text
.
|-- app.py
|-- train_model.py
|-- utils.py
|-- requirements.txt
`-- model/
    `-- model_summary.json
```

## Getting Started

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the app

```powershell
streamlit run app.py
```

The app fetches live data on startup and caches the response for 12 hours unless you manually refresh it in the UI.

## Rebuild the Model Summary

To regenerate the summary artifact saved in `model/model_summary.json`:

```powershell
python train_model.py
```

## Data Source

- World Bank Indicators API: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

## Notes

- This project is intended for scenario exploration, not as an official census or policy forecasting system.
- Because the app depends on live API data, outputs may change over time as the World Bank updates its datasets.
