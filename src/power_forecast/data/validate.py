from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ValidationCheck:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


def _check_status(checks: list[ValidationCheck]) -> str:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "pass"


def _add_check(
    checks: list[ValidationCheck],
    name: str,
    passed: bool,
    details: dict[str, Any] | None = None,
    warn_only: bool = False,
) -> None:
    if passed:
        status = "pass"
    else:
        status = "warn" if warn_only else "fail"

    checks.append(
        ValidationCheck(
            name=name,
            status=status,
            details=details or {},
        )
    )


def _validate_file_exists(path: Path, dataset_name: str) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    _add_check(
        checks,
        name=f"{dataset_name}_file_exists",
        passed=path.exists(),
        details={"path": str(path)},
    )
    return checks


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    missing = [col for col in required_columns if col not in df.columns]
    _add_check(
        checks,
        name=f"{dataset_name}_required_columns",
        passed=len(missing) == 0,
        details={
            "required_columns": required_columns,
            "actual_columns": list(df.columns),
            "missing_columns": missing,
        },
    )
    return checks


def _validate_min_rows(
    df: pd.DataFrame,
    min_rows: int,
    dataset_name: str,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    _add_check(
        checks,
        name=f"{dataset_name}_min_rows",
        passed=len(df) >= min_rows,
        details={
            "actual_rows": int(len(df)),
            "min_rows": int(min_rows),
        },
    )
    return checks


def _validate_missing_rate(
    df: pd.DataFrame,
    columns: list[str],
    max_missing_rate: float,
    dataset_name: str,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []

    missing_rates = {}
    failing_columns = []

    for col in columns:
        if col not in df.columns:
            continue
        rate = float(df[col].isna().mean())
        missing_rates[col] = rate
        if rate > max_missing_rate:
            failing_columns.append(col)

    _add_check(
        checks,
        name=f"{dataset_name}_missing_rate",
        passed=len(failing_columns) == 0,
        details={
            "max_missing_rate": max_missing_rate,
            "missing_rates": missing_rates,
            "failing_columns": failing_columns,
        },
    )
    return checks


def _validate_timestamps(
    df: pd.DataFrame,
    timestamp_column: str,
    dataset_name: str,
) -> tuple[list[ValidationCheck], pd.Series | None]:
    checks: list[ValidationCheck] = []

    if timestamp_column not in df.columns:
        _add_check(
            checks,
            name=f"{dataset_name}_timestamp_column_exists",
            passed=False,
            details={"timestamp_column": timestamp_column},
        )
        return checks, None

    ts = pd.to_datetime(df[timestamp_column], errors="coerce")

    parse_failures = int(ts.isna().sum())
    _add_check(
        checks,
        name=f"{dataset_name}_timestamps_parse",
        passed=parse_failures == 0,
        details={"parse_failures": parse_failures},
    )

    if parse_failures > 0:
        return checks, ts

    duplicate_count = int(ts.duplicated().sum())
    _add_check(
        checks,
        name=f"{dataset_name}_no_duplicate_timestamps",
        passed=duplicate_count == 0,
        details={"duplicate_count": duplicate_count},
    )

    is_monotonic = bool(ts.is_monotonic_increasing)
    _add_check(
        checks,
        name=f"{dataset_name}_timestamps_monotonic_increasing",
        passed=is_monotonic,
        details={
            "min_timestamp": str(ts.min()) if len(ts) else None,
            "max_timestamp": str(ts.max()) if len(ts) else None,
        },
    )

    if len(ts) >= 2:
        full_range = pd.date_range(start=ts.min(), end=ts.max(), freq="h")
        observed_unique = pd.Index(ts.drop_duplicates())
        missing_hours = full_range.difference(observed_unique)

        # Some APIs include one extra endpoint or have local DST effects.
        # For raw validation, small gaps are a warning, large gaps are a failure.
        missing_count = int(len(missing_hours))
        expected_hours = int(len(full_range))
        actual_unique_hours = int(len(observed_unique))

        gap_rate = missing_count / max(expected_hours, 1)
        passed = gap_rate <= 0.01

        _add_check(
            checks,
            name=f"{dataset_name}_hourly_coverage",
            passed=passed,
            warn_only=missing_count > 0 and gap_rate <= 0.05,
            details={
                "expected_hours": expected_hours,
                "actual_unique_hours": actual_unique_hours,
                "missing_hours": missing_count,
                "gap_rate": gap_rate,
                "first_missing_examples": [str(x) for x in missing_hours[:5]],
            },
        )

    return checks, ts


def _validate_numeric_ranges(
    df: pd.DataFrame,
    value_ranges: dict[str, dict[str, float]],
    dataset_name: str,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []

    for col, bounds in value_ranges.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        lower = bounds.get("min")
        upper = bounds.get("max")

        too_low = int((values < lower).sum()) if lower is not None else 0
        too_high = int((values > upper).sum()) if upper is not None else 0
        non_numeric = int(values.isna().sum() - df[col].isna().sum())

        passed = too_low == 0 and too_high == 0 and non_numeric == 0

        _add_check(
            checks,
            name=f"{dataset_name}_{col}_range",
            passed=passed,
            details={
                "column": col,
                "min_allowed": lower,
                "max_allowed": upper,
                "observed_min": float(values.min()) if values.notna().any() else None,
                "observed_max": float(values.max()) if values.notna().any() else None,
                "too_low_count": too_low,
                "too_high_count": too_high,
                "non_numeric_count": non_numeric,
            },
        )

    return checks


def validate_dataset(
    path: str | Path,
    dataset_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    checks = _validate_file_exists(path, dataset_name)

    if not path.exists():
        return {
            "dataset": dataset_name,
            "path": str(path),
            "status": _check_status(checks),
            "checks": [c.__dict__ for c in checks],
        }

    df = pd.read_csv(path)

    checks.extend(
        _validate_required_columns(
            df=df,
            required_columns=list(config["required_columns"]),
            dataset_name=dataset_name,
        )
    )

    checks.extend(
        _validate_min_rows(
            df=df,
            min_rows=int(config["min_rows"]),
            dataset_name=dataset_name,
        )
    )

    columns_for_missing = list(
        set(config["required_columns"]) | set(config.get("numeric_columns", []))
    )
    checks.extend(
        _validate_missing_rate(
            df=df,
            columns=columns_for_missing,
            max_missing_rate=float(config["max_missing_rate"]),
            dataset_name=dataset_name,
        )
    )

    timestamp_checks, _ = _validate_timestamps(
        df=df,
        timestamp_column=str(config["timestamp_column"]),
        dataset_name=dataset_name,
    )
    checks.extend(timestamp_checks)

    checks.extend(
        _validate_numeric_ranges(
            df=df,
            value_ranges=dict(config.get("value_ranges", {})),
            dataset_name=dataset_name,
        )
    )

    return {
        "dataset": dataset_name,
        "path": str(path),
        "status": _check_status(checks),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "checks": [c.__dict__ for c in checks],
    }


def write_validation_report(
    output_path: str | Path,
    report: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
