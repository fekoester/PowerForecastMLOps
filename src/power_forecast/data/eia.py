from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


EIA_REGION_DATA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def fetch_eia_hourly_demand(
    respondent: str,
    demand_type: str,
    lookback_days: int,
    output_path: str | Path,
) -> pd.DataFrame:
    """Fetch hourly electricity demand from the EIA v2 API.

    Parameters
    ----------
    respondent:
        EIA balancing authority / region code, e.g. CISO.
    demand_type:
        Usually "D" for demand.
    lookback_days:
        Number of days back from now to request.
    output_path:
        CSV path where raw demand data is stored.
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "EIA_API_KEY is not set. Create a .env file or export EIA_API_KEY in your shell."
        )

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=lookback_days)

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": demand_type,
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }

    response = requests.get(EIA_REGION_DATA_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    rows = payload.get("response", {}).get("data", [])
    if not rows:
        raise RuntimeError(f"EIA returned no data. Payload keys: {payload.keys()}")

    df = pd.DataFrame(rows)

    # Normalize expected fields.
    # EIA usually returns: period, respondent, respondent-name, type, type-name, value, value-units
    df = df.rename(
        columns={
            "period": "timestamp_utc",
            "value": "demand_mwh",
            "respondent": "region",
            "type": "demand_type",
        }
    )

    required = ["timestamp_utc", "demand_mwh", "region", "demand_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"EIA response missing required columns: {missing}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
