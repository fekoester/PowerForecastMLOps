from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from power_forecast.utils.env import require_env_var

import pandas as pd
import requests


EIA_REGION_DATA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def fetch_eia_hourly_demand(
    respondent: str,
    demand_type: str,
    lookback_days: int,
    output_path: str | Path,
) -> pd.DataFrame:
    """Fetch hourly electricity demand from the EIA v2 API with pagination.

    The EIA API returns at most `length` rows per request. For multi-year
    hourly data, we need to request multiple pages with increasing offset.
    """
    api_key = require_env_var("EIA_API_KEY")

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=lookback_days)

    page_size = 5000
    offset = 0
    all_rows: list[dict] = []

    while True:
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
            "offset": offset,
            "length": page_size,
        }

        response = requests.get(EIA_REGION_DATA_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("response", {}).get("data", [])
        if not rows and offset == 0:
            raise RuntimeError(f"EIA returned no data. Payload keys: {payload.keys()}")

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        offset += page_size

    df = pd.DataFrame(all_rows)

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

    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)

    # Extra safety: keep only the requested interval.
    df = df[
        (df["timestamp_utc"] >= pd.Timestamp(start))
        & (df["timestamp_utc"] <= pd.Timestamp(end))
    ].reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
