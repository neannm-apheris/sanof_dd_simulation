#!/usr/bin/env python3
"""Compute regression metrics from a Chemprop prediction CSV and truth CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--row-id-column", default="_row_id")
    parser.add_argument("--target-columns", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-through-fold", required=True, type=int)
    parser.add_argument("--evaluation-scope", required=True)
    return parser.parse_args()


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if not len(y_true):
        raise ValueError("No finite truth/prediction pairs were found")

    error = y_pred - y_true
    abs_error = np.abs(error)
    squared_error = error**2
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(np.divide(2 * abs_error, denominator, out=np.zeros_like(error), where=denominator != 0))

    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = stats.pearsonr(y_true, y_pred).statistic
        spearman = stats.spearmanr(y_true, y_pred).statistic
        kendall = stats.kendalltau(y_true, y_pred).statistic
        slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
    else:
        pearson = spearman = kendall = slope = intercept = r2 = explained_variance = float("nan")

    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    covariance = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    ccc_denominator = np.var(y_true) + np.var(y_pred) + (true_mean - pred_mean) ** 2
    ccc = 2 * covariance / ccc_denominator if ccc_denominator else float("nan")

    return {
        "n": int(len(y_true)),
        "mae": finite_or_none(np.mean(abs_error)),
        "median_absolute_error": finite_or_none(np.median(abs_error)),
        "rmse": finite_or_none(np.sqrt(np.mean(squared_error))),
        "mse": finite_or_none(np.mean(squared_error)),
        "r2": finite_or_none(r2),
        "explained_variance": finite_or_none(explained_variance),
        "pearson_r": finite_or_none(pearson),
        "spearman_rho": finite_or_none(spearman),
        "kendall_tau": finite_or_none(kendall),
        "concordance_correlation_coefficient": finite_or_none(ccc),
        "mean_error_bias": finite_or_none(np.mean(error)),
        "error_std": finite_or_none(np.std(error, ddof=1) if len(error) > 1 else 0.0),
        "max_absolute_error": finite_or_none(np.max(abs_error)),
        "smape": finite_or_none(smape),
        "regression_slope": finite_or_none(slope),
        "regression_intercept": finite_or_none(intercept),
        "true_mean": finite_or_none(true_mean),
        "prediction_mean": finite_or_none(pred_mean),
        "true_std": finite_or_none(np.std(y_true, ddof=1) if len(y_true) > 1 else 0.0),
        "prediction_std": finite_or_none(np.std(y_pred, ddof=1) if len(y_pred) > 1 else 0.0),
    }


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth)
    predictions = pd.read_csv(args.predictions)
    key = args.row_id_column

    for frame_name, frame in (("truth", truth), ("predictions", predictions)):
        if key not in frame:
            raise ValueError(f"{frame_name} file has no {key!r} column")
        if frame[key].duplicated().any():
            raise ValueError(f"{frame_name} file contains duplicate {key!r} values")
    missing_targets = [column for column in args.target_columns if column not in predictions]
    if missing_targets:
        raise ValueError(f"Prediction output is missing target columns: {missing_targets}")

    truth_columns = [key, "DATE", "CPD_ID", "SERIES_ID", *args.target_columns]
    merged = truth[truth_columns].merge(
        predictions[[key, *args.target_columns]],
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("_actual", "_predicted"),
    )
    if len(merged) != len(truth) or len(merged) != len(predictions):
        raise ValueError(
            f"Row mismatch after joining truth ({len(truth)}) and predictions ({len(predictions)}): "
            f"only {len(merged)} rows matched"
        )

    records: list[dict[str, object]] = []
    for target in args.target_columns:
        result = regression_metrics(
            merged[f"{target}_actual"].to_numpy(dtype=float),
            merged[f"{target}_predicted"].to_numpy(dtype=float),
        )
        records.append(
            {
                "model": args.model,
                "train_through_fold": args.train_through_fold,
                "evaluation_scope": args.evaluation_scope,
                "target": target,
                **result,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
