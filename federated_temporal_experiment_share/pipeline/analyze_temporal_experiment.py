#!/usr/bin/env python3
"""Analyze completed replicated rolling-origin Chemprop predictions.

Inference follows the supplied practical method-comparison workflow: each
rolling origin is an independent modelling challenge; 25 crossed model fits are
the repeated observations; full-size paired bootstrap draws quantify uncertainty
but are never treated as independent samples.  Repeated-measures ANOVA/Tukey is
reported alongside Friedman/Conover-Holm, with diagnostics selecting the stated
primary result.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from evaluate_predictions import regression_metrics


BOOTSTRAP_METRICS = ("rmse", "mae", "r2")
LOWER_IS_BETTER = {"rmse", "mae", "mse", "median_absolute_error", "smape", "max_absolute_error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("temporal_experiment"))
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--practical-threshold-percent", type=float, default=None)
    return parser.parse_args()


def float_or_nan(value: object) -> float:
    return np.nan if value is None else float(value)


def parse_replicate_id(replicate_id: str) -> tuple[int, int]:
    left, right = replicate_id.split("__")
    return int(left.removeprefix("data_")), int(right.removeprefix("torch_"))


def load_prediction(
    truth: pd.DataFrame, path: Path, targets: list[str]
) -> pd.DataFrame:
    prediction = pd.read_csv(path)
    required = ["_row_id", *targets]
    missing = [column for column in required if column not in prediction.columns]
    if missing:
        raise ValueError(f"{path} is missing prediction columns {missing}")
    if prediction["_row_id"].duplicated().any():
        raise ValueError(f"Duplicate _row_id values in {path}")
    joined = truth[["_row_id", "temporal_fold", *targets]].merge(
        prediction[required],
        on="_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_actual", "_predicted"),
    )
    if joined[[f"{target}_predicted" for target in targets]].isna().any().any():
        raise ValueError(f"Predictions in {path} do not cover every future row")
    return joined


def scope_frame(joined: pd.DataFrame, scope: str, origin: int, last_fold: int) -> pd.DataFrame:
    if scope == "next_fold":
        return joined.loc[joined.temporal_fold == origin + 1].copy()
    if scope == "all_future":
        return joined.copy()
    if scope == "last_fold":
        return joined.loc[joined.temporal_fold == last_fold].copy()
    raise ValueError(scope)


def collect_predictions(
    output_dir: Path, manifest: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = list(manifest["model_targets"])
    strategies = list(manifest["strategies"])
    n_folds = int(manifest["n_folds"])
    metric_records: list[dict[str, object]] = []
    point_frames: list[pd.DataFrame] = []
    for origin in manifest["rolling_origins"]:
        origin = int(origin)
        origin_dir = output_dir / f"origin_{origin:02d}"
        truth = pd.read_csv(origin_dir / "evaluation" / "all_future_truth.csv")
        for replicate_dir in sorted((origin_dir / "replicates").glob("data_*__torch_*")):
            replicate_id = replicate_dir.name
            data_seed, pytorch_seed = parse_replicate_id(replicate_id)
            for strategy in strategies:
                path = replicate_dir / "predictions" / f"{strategy}_all_future.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                joined = load_prediction(truth, path, targets)
                for scope in ("next_fold", "all_future", "last_fold"):
                    scoped = scope_frame(joined, scope, origin, n_folds)
                    for target in targets:
                        y_true = scoped[f"{target}_actual"].to_numpy(dtype=float)
                        y_pred = scoped[f"{target}_predicted"].to_numpy(dtype=float)
                        values = regression_metrics(y_true, y_pred)
                        metric_records.append(
                            {
                                "origin": origin,
                                "evaluation_scope": scope,
                                "target": target,
                                "strategy": strategy,
                                "replicate_id": replicate_id,
                                "data_seed": data_seed,
                                "pytorch_seed": pytorch_seed,
                                **{key: float_or_nan(value) for key, value in values.items()},
                            }
                        )
                    points = scoped[["_row_id", "temporal_fold"]].copy()
                    points["origin"] = origin
                    points["evaluation_scope"] = scope
                    points["strategy"] = strategy
                    points["replicate_id"] = replicate_id
                    points["data_seed"] = data_seed
                    points["pytorch_seed"] = pytorch_seed
                    for target in targets:
                        points[f"{target}__actual"] = scoped[f"{target}_actual"].to_numpy()
                        points[f"{target}__predicted"] = scoped[f"{target}_predicted"].to_numpy()
                    point_frames.append(points)
    return pd.DataFrame(metric_records), pd.concat(point_frames, ignore_index=True)


def bootstrap_metric_values(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - y_true[None, :]
    mse = np.mean(errors**2, axis=1)
    mae = np.mean(np.abs(errors), axis=1)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.full(len(predictions), np.nan) if denominator == 0 else 1.0 - np.sum(errors**2, axis=1) / denominator
    return {"rmse": float(np.mean(np.sqrt(mse))), "mae": float(np.mean(mae)), "r2": float(np.mean(r2))}


def bootstrap_distributions(
    points: pd.DataFrame,
    manifest: dict[str, object],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    strategies = list(manifest["strategies"])
    data_seeds = list(manifest["data_seeds"])
    pytorch_seeds = list(manifest["pytorch_seeds"])
    targets = list(manifest["model_targets"])
    replicate_order = [f"data_{d}__torch_{p}" for d in data_seeds for p in pytorch_seeds]
    replicate_lookup = {(d, p): i for i, (d, p) in enumerate(itertools.product(data_seeds, pytorch_seeds))}
    records: list[dict[str, object]] = []
    groups = itertools.product(
        [int(x) for x in manifest["rolling_origins"]],
        ("next_fold", "all_future", "last_fold"),
        targets,
    )
    seed_sequence = np.random.SeedSequence(bootstrap_seed)
    group_seeds = seed_sequence.spawn(len(list(manifest["rolling_origins"])) * 3 * len(targets))
    for group_number, (origin, scope, target) in enumerate(groups):
        subset = points.loc[(points.origin == origin) & (points.evaluation_scope == scope)]
        row_ids = np.sort(subset._row_id.unique())
        truth_series = (
            subset.loc[subset.strategy == strategies[0], ["_row_id", f"{target}__actual"]]
            .drop_duplicates("_row_id")
            .set_index("_row_id")
            .loc[row_ids, f"{target}__actual"]
        )
        y_true = truth_series.to_numpy(dtype=float)
        arrays: dict[str, np.ndarray] = {}
        for strategy in strategies:
            pivot = subset.loc[subset.strategy == strategy].pivot(
                index="replicate_id", columns="_row_id", values=f"{target}__predicted"
            )
            arrays[strategy] = pivot.loc[replicate_order, row_ids].to_numpy(dtype=float)
        rng = np.random.default_rng(group_seeds[group_number])
        for draw in range(n_bootstrap):
            row_indices = rng.integers(0, len(row_ids), size=len(row_ids))
            sampled_data = rng.choice(data_seeds, size=len(data_seeds), replace=True)
            sampled_torch = rng.choice(pytorch_seeds, size=len(pytorch_seeds), replace=True)
            replicate_indices = [replicate_lookup[(int(d), int(p))] for d in sampled_data for p in sampled_torch]
            sampled_truth = y_true[row_indices]
            for strategy in strategies:
                sampled_predictions = arrays[strategy][replicate_indices][:, row_indices]
                values = bootstrap_metric_values(sampled_truth, sampled_predictions)
                records.append(
                    {
                        "origin": origin,
                        "evaluation_scope": scope,
                        "target": target,
                        "strategy": strategy,
                        "bootstrap_draw": draw,
                        **values,
                    }
                )
    return pd.DataFrame(records)


def bootstrap_summaries(
    boot: pd.DataFrame,
    full_metrics: pd.DataFrame,
    practical_threshold: float,
    strategy_comparators: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []
    keys = ["origin", "evaluation_scope", "target", "strategy"]
    for key, group in boot.groupby(keys, sort=False):
        central = full_metrics.loc[
            (full_metrics.origin == key[0])
            & (full_metrics.evaluation_scope == key[1])
            & (full_metrics.target == key[2])
            & (full_metrics.strategy == key[3])
        ]
        for metric in BOOTSTRAP_METRICS:
            values = group[metric].to_numpy(dtype=float)
            summary_records.append(
                dict(
                    zip(keys, key),
                    metric=metric,
                    estimate=float(central[metric].mean()),
                    ci_low=float(np.quantile(values, 0.025)),
                    ci_high=float(np.quantile(values, 0.975)),
                    n_model_fits=len(central),
                    n_bootstrap=len(values),
                )
            )
    index = ["origin", "evaluation_scope", "target", "bootstrap_draw"]
    for metric in BOOTSTRAP_METRICS:
        wide = boot.pivot(index=index, columns="strategy", values=metric).reset_index()
        for strategy, comparator in strategy_comparators.items():
            if strategy not in wide.columns or comparator not in wide.columns:
                continue
            if metric in LOWER_IS_BETTER:
                raw = wide[comparator] - wide[strategy]
            else:
                raw = wide[strategy] - wide[comparator]
            relative = 100.0 * raw / wide[comparator].abs().clip(lower=np.finfo(float).eps)
            temp = wide[index].copy()
            temp["benefit"] = raw
            temp["relative_benefit_percent"] = relative
            for key, group in temp.groupby(index[:-1], sort=False):
                rel = group.relative_benefit_percent.to_numpy(dtype=float)
                benefit = group.benefit.to_numpy(dtype=float)
                pair_records.append(
                    {
                        "origin": key[0],
                        "evaluation_scope": key[1],
                        "target": key[2],
                        "metric": metric,
                        "strategy": strategy,
                        "baseline_strategy": comparator,
                        "comparison": f"{strategy}_vs_{comparator}",
                        "benefit_estimate": float(np.mean(benefit)),
                        "benefit_ci_low": float(np.quantile(benefit, 0.025)),
                        "benefit_ci_high": float(np.quantile(benefit, 0.975)),
                        "relative_benefit_percent": float(np.mean(rel)),
                        "relative_ci_low": float(np.quantile(rel, 0.025)),
                        "relative_ci_high": float(np.quantile(rel, 0.975)),
                        "probability_better": float(np.mean(rel > 0)),
                        "probability_practically_better": float(np.mean(rel > practical_threshold)),
                        "practical_threshold_percent": practical_threshold,
                    }
                )
    return pd.DataFrame(summary_records), pd.DataFrame(pair_records)


def repeated_measures_anova(matrix: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    values = matrix.to_numpy(dtype=float)
    n, k = values.shape
    grand = values.mean()
    subject_means = values.mean(axis=1, keepdims=True)
    method_means = values.mean(axis=0, keepdims=True)
    residuals = values - subject_means - method_means + grand
    ss_method = n * np.sum((method_means - grand) ** 2)
    ss_error = np.sum(residuals**2)
    df_method = k - 1
    df_error = (n - 1) * (k - 1)
    ms_method = ss_method / df_method
    ms_error = ss_error / df_error
    f_value = ms_method / ms_error if ms_error > 0 else np.inf
    p_value = float(stats.f.sf(f_value, df_method, df_error))
    partial_eta_squared = float(ss_method / (ss_method + ss_error)) if ss_method + ss_error else np.nan
    return {
        "statistic": float(f_value),
        "p_value": p_value,
        "df_1": df_method,
        "df_2": df_error,
        "error_mean_square": float(ms_error),
        "partial_eta_squared": partial_eta_squared,
    }, residuals.ravel()


def tukey_from_repeated_anova(matrix: pd.DataFrame, mse: float, df_error: int) -> list[dict[str, object]]:
    n, k = matrix.shape
    se = math.sqrt(mse / n)
    q_critical = float(stats.studentized_range.ppf(0.95, k, df_error))
    records: list[dict[str, object]] = []
    for a, b in itertools.combinations(matrix.columns, 2):
        difference = float(matrix[a].mean() - matrix[b].mean())
        q_value = abs(difference) / se if se else np.inf
        records.append(
            {
                "method_a": a,
                "method_b": b,
                "mean_a_minus_b": difference,
                "ci_low": difference - q_critical * se,
                "ci_high": difference + q_critical * se,
                "p_adjusted": float(stats.studentized_range.sf(q_value, k, df_error)),
                "effect_size_dz": float((matrix[a] - matrix[b]).mean() / (matrix[a] - matrix[b]).std(ddof=1)),
                "test": "repeated_measures_Tukey_HSD",
            }
        )
    return records


def conover_friedman_holm(matrix: pd.DataFrame) -> list[dict[str, object]]:
    values = matrix.to_numpy(dtype=float)
    ranks = np.apply_along_axis(stats.rankdata, 1, values)
    n, k = ranks.shape
    rank_sums = ranks.sum(axis=0)
    a1 = np.sum(ranks**2)
    s2 = (a1 - k * n * ((k + 1.0) ** 2.0) / 4.0) / (k - 1.0)
    t2 = np.sum((rank_sums - n * (k + 1.0) / 2.0) ** 2.0) / s2
    a_value = s2 * (2.0 * n * (k - 1.0)) / (n * k - k - n + 1.0)
    b_value = 1.0 - t2 / (n * (k - 1.0))
    denominator = math.sqrt(max(a_value * b_value, 0.0))
    df = n * k - k - n + 1
    raw_records: list[dict[str, object]] = []
    raw_p: list[float] = []
    for i, j in itertools.combinations(range(k), 2):
        statistic = abs(rank_sums[i] - rank_sums[j]) / denominator if denominator else np.inf
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=df))
        raw_p.append(p_value)
        raw_records.append(
            {
                "method_a": matrix.columns[i],
                "method_b": matrix.columns[j],
                "mean_a_minus_b": float(matrix.iloc[:, i].mean() - matrix.iloc[:, j].mean()),
                "statistic": float(statistic),
                "p_unadjusted": p_value,
                "effect_size_dz": float(
                    (matrix.iloc[:, i] - matrix.iloc[:, j]).mean()
                    / (matrix.iloc[:, i] - matrix.iloc[:, j]).std(ddof=1)
                ),
                "test": "Conover_Friedman_Holm",
            }
        )
    adjusted = multipletests(raw_p, method="holm")[1]
    for record, value in zip(raw_records, adjusted):
        record["p_adjusted"] = float(value)
    return raw_records


def analyse_matrix(matrix: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    anova, residuals = repeated_measures_anova(matrix)
    shapiro_p = float(stats.shapiro(residuals).pvalue) if len(residuals) <= 5000 else np.nan
    variances = matrix.var(ddof=1)
    positive = variances[variances > 0]
    variance_ratio = float(positive.max() / positive.min()) if len(positive) else np.nan
    parametric_ok = bool((np.isnan(shapiro_p) or shapiro_p >= 0.001) and (np.isnan(variance_ratio) or variance_ratio <= 9.0))
    friedman = stats.friedmanchisquare(*(matrix[column].to_numpy() for column in matrix.columns))
    omnibus = [
        {
            "test": "repeated_measures_ANOVA",
            **anova,
            "shapiro_residual_p": shapiro_p,
            "variance_ratio": variance_ratio,
            "parametric_assumptions_acceptable": parametric_ok,
            "recommended_test": "repeated_measures_ANOVA" if parametric_ok else "Friedman",
        },
        {
            "test": "Friedman",
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
            "df_1": matrix.shape[1] - 1,
            "df_2": np.nan,
            "error_mean_square": np.nan,
            "partial_eta_squared": np.nan,
            "shapiro_residual_p": shapiro_p,
            "variance_ratio": variance_ratio,
            "parametric_assumptions_acceptable": parametric_ok,
            "recommended_test": "repeated_measures_ANOVA" if parametric_ok else "Friedman",
        },
    ]
    pairwise = tukey_from_repeated_anova(matrix, anova["error_mean_square"], int(anova["df_2"]))
    pairwise.extend(conover_friedman_holm(matrix))
    for record in pairwise:
        record["recommended"] = (
            record["test"] == "repeated_measures_Tukey_HSD" if parametric_ok else record["test"] == "Conover_Friedman_Holm"
        )
    return omnibus, pairwise, residuals


def inferential_tables(
    full_metrics: pd.DataFrame, strategy_families: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    omnibus_records: list[dict[str, object]] = []
    pairwise_records: list[dict[str, object]] = []
    residual_records: list[dict[str, object]] = []
    grouping = ["origin", "evaluation_scope", "target"]
    for comparison_family, strategies in strategy_families.items():
        strategies = [strategy for strategy in strategies if strategy in set(full_metrics.strategy)]
        if len(strategies) < 2:
            continue
        family_metrics = full_metrics.loc[full_metrics.strategy.isin(strategies)]
        for metric in BOOTSTRAP_METRICS:
            for key, group in family_metrics.groupby(grouping, sort=False):
                matrix = group.pivot(index="replicate_id", columns="strategy", values=metric).dropna()
                if matrix.shape[1] < 2:
                    continue
                omnibus, pairwise, residual_values = analyse_matrix(matrix)
                context = {
                    "comparison_family": comparison_family,
                    "analysis_level": "per_origin",
                    "origin": key[0],
                    "evaluation_scope": key[1],
                    "target": key[2],
                    "metric": metric,
                }
                omnibus_records.extend(
                    [{**context, **record, "n_blocks": len(matrix)} for record in omnibus]
                )
                pairwise_records.extend(
                    [{**context, **record, "n_blocks": len(matrix)} for record in pairwise]
                )
                residual_records.extend(
                    [{**context, "residual": float(value)} for value in residual_values]
                )

            averaged = (
                family_metrics.groupby(
                    ["evaluation_scope", "target", "replicate_id", "strategy"], as_index=False
                )[metric]
                .mean()
            )
            for key, group in averaged.groupby(["evaluation_scope", "target"], sort=False):
                matrix = group.pivot(index="replicate_id", columns="strategy", values=metric).dropna()
                if matrix.shape[1] < 2:
                    continue
                omnibus, pairwise, residual_values = analyse_matrix(matrix)
                context = {
                    "comparison_family": comparison_family,
                    "analysis_level": "all_origins_equal_weight",
                    "origin": "all",
                    "evaluation_scope": key[0],
                    "target": key[1],
                    "metric": metric,
                }
                omnibus_records.extend(
                    [{**context, **record, "n_blocks": len(matrix)} for record in omnibus]
                )
                pairwise_records.extend(
                    [{**context, **record, "n_blocks": len(matrix)} for record in pairwise]
                )
                residual_records.extend(
                    [{**context, "residual": float(value)} for value in residual_values]
                )
    return pd.DataFrame(omnibus_records), pd.DataFrame(pairwise_records), pd.DataFrame(residual_records)


def add_significance_labels(
    pair_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    practical_threshold: float,
    strategy_comparators: dict[str, str],
    strategy_primary_family: dict[str, str],
) -> pd.DataFrame:
    selected_tests: list[pd.DataFrame] = []
    for strategy, baseline in strategy_comparators.items():
        family = strategy_primary_family.get(strategy, "prospective_multitask")
        selected = pairwise.loc[
            pairwise.recommended
            & (pairwise.analysis_level == "per_origin")
            & (pairwise.comparison_family == family)
            & (
                ((pairwise.method_a == baseline) & (pairwise.method_b == strategy))
                | ((pairwise.method_b == baseline) & (pairwise.method_a == strategy))
            )
        ].copy()
        selected["strategy"] = strategy
        selected["baseline_strategy"] = baseline
        selected_tests.append(selected)
    recommended = pd.concat(selected_tests, ignore_index=True) if selected_tests else pd.DataFrame()
    keys = [
        "origin",
        "evaluation_scope",
        "target",
        "metric",
        "strategy",
        "baseline_strategy",
    ]
    test_columns = keys + ["comparison_family", "test", "p_adjusted"]
    merged = pair_summary.merge(recommended[test_columns], on=keys, how="left")
    statistically_better = (merged.p_adjusted < 0.05) & (merged.relative_ci_low > 0)
    statistically_worse = (merged.p_adjusted < 0.05) & (merged.relative_ci_high < 0)
    fixed_test_better_only = (
        (merged.p_adjusted < 0.05)
        & (merged.relative_benefit_percent > 0)
        & (merged.relative_ci_low <= 0)
    )
    fixed_test_worse_only = (
        (merged.p_adjusted < 0.05)
        & (merged.relative_benefit_percent < 0)
        & (merged.relative_ci_high >= 0)
    )
    practically_better = merged.relative_ci_low > practical_threshold
    practically_worse = merged.relative_ci_high < -practical_threshold
    merged["conclusion"] = np.select(
        [
            statistically_better & practically_better,
            statistically_better,
            statistically_worse & practically_worse,
            statistically_worse,
            fixed_test_better_only,
            fixed_test_worse_only,
        ],
        [
            "statistically and practically better",
            "statistically better; practical threshold not established",
            "statistically and practically worse",
            "statistically worse; practical threshold not established",
            "fixed-test difference favours federation; joint bootstrap includes zero",
            "fixed-test difference favours scratch; joint bootstrap includes zero",
        ],
        default="no statistically distinct performance",
    )
    return merged


def global_bootstrap_summary(
    boot: pd.DataFrame,
    practical_threshold: float,
    strategy_comparators: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric in BOOTSTRAP_METRICS:
        averaged = (
            boot.groupby(["evaluation_scope", "target", "strategy", "bootstrap_draw"], as_index=False)[metric]
            .mean()
        )
        wide = averaged.pivot(
            index=["evaluation_scope", "target", "bootstrap_draw"], columns="strategy", values=metric
        ).reset_index()
        for strategy, comparator in strategy_comparators.items():
            if strategy not in wide.columns or comparator not in wide.columns:
                continue
            raw = wide[comparator] - wide[strategy] if metric in LOWER_IS_BETTER else wide[strategy] - wide[comparator]
            relative = 100 * raw / wide[comparator].abs().clip(lower=np.finfo(float).eps)
            temp = wide[["evaluation_scope", "target"]].copy()
            temp["benefit"] = raw
            temp["relative"] = relative
            for key, group in temp.groupby(["evaluation_scope", "target"], sort=False):
                values = group.relative.to_numpy()
                records.append(
                    {
                        "analysis_level": "all_origins_equal_weight",
                        "evaluation_scope": key[0],
                        "target": key[1],
                        "metric": metric,
                        "strategy": strategy,
                        "baseline_strategy": comparator,
                        "relative_benefit_percent": float(np.mean(values)),
                        "relative_ci_low": float(np.quantile(values, 0.025)),
                        "relative_ci_high": float(np.quantile(values, 0.975)),
                        "probability_better": float(np.mean(values > 0)),
                        "probability_practically_better": float(np.mean(values > practical_threshold)),
                        "practical_threshold_percent": practical_threshold,
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    n_bootstrap = args.bootstrap_samples or int(manifest["bootstrap_samples"])
    bootstrap_seed = args.bootstrap_seed or int(manifest["bootstrap_seed"])
    practical_threshold = (
        args.practical_threshold_percent
        if args.practical_threshold_percent is not None
        else float(manifest["practical_threshold_percent"])
    )
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    full_metrics, points = collect_predictions(output_dir, manifest)
    full_metrics.to_csv(analysis_dir / "full_metrics.csv", index=False)
    points.to_csv(analysis_dir / "prediction_points.csv.gz", index=False, compression="gzip")
    boot = bootstrap_distributions(points, manifest, n_bootstrap, bootstrap_seed)
    boot.to_csv(analysis_dir / "bootstrap_metrics.csv.gz", index=False, compression="gzip")
    strategies = list(manifest["strategies"])
    strategy_comparators = dict(
        manifest.get(
            "strategy_comparators",
            {strategy: "scratch" for strategy in strategies if strategy != "scratch"},
        )
    )
    strategy_families = dict(
        manifest.get("strategy_families", {"prospective_multitask": strategies})
    )
    strategy_primary_family = dict(
        manifest.get(
            "strategy_primary_family",
            {strategy: "prospective_multitask" for strategy in strategy_comparators},
        )
    )
    bootstrap_summary, pair_summary = bootstrap_summaries(
        boot, full_metrics, practical_threshold, strategy_comparators
    )
    bootstrap_summary.to_csv(analysis_dir / "bootstrap_summary.csv", index=False)
    omnibus, pairwise, residuals = inferential_tables(full_metrics, strategy_families)
    omnibus.to_csv(analysis_dir / "omnibus_tests.csv", index=False)
    pairwise.to_csv(analysis_dir / "pairwise_tests.csv", index=False)
    residuals.to_csv(analysis_dir / "anova_residuals.csv", index=False)
    significance = add_significance_labels(
        pair_summary,
        pairwise,
        practical_threshold,
        strategy_comparators,
        strategy_primary_family,
    )
    significance.to_csv(analysis_dir / "per_origin_significance.csv", index=False)
    pair_summary.to_csv(analysis_dir / "pairwise_bootstrap_summary.csv", index=False)
    global_summary = global_bootstrap_summary(
        boot, practical_threshold, strategy_comparators
    )
    global_pairwise_frames: list[pd.DataFrame] = []
    for strategy, baseline in strategy_comparators.items():
        family = strategy_primary_family.get(strategy, "prospective_multitask")
        selected = pairwise.loc[
            (pairwise.analysis_level == "all_origins_equal_weight")
            & pairwise.recommended
            & (pairwise.comparison_family == family)
            & (
                ((pairwise.method_a == baseline) & (pairwise.method_b == strategy))
                | ((pairwise.method_b == baseline) & (pairwise.method_a == strategy))
            )
        ].copy()
        selected["strategy"] = strategy
        selected["baseline_strategy"] = baseline
        global_pairwise_frames.append(selected)
    global_pairwise = pd.concat(global_pairwise_frames, ignore_index=True)
    merge_keys = [
        "analysis_level",
        "evaluation_scope",
        "target",
        "metric",
        "strategy",
        "baseline_strategy",
    ]
    global_summary = global_summary.merge(
        global_pairwise[merge_keys + ["comparison_family", "test", "p_adjusted"]],
        on=merge_keys,
        how="left",
    )
    global_summary["conclusion"] = np.select(
        [
            (global_summary.p_adjusted < 0.05) & (global_summary.relative_ci_low > practical_threshold),
            (global_summary.p_adjusted < 0.05) & (global_summary.relative_ci_low > 0),
            (global_summary.p_adjusted < 0.05) & (global_summary.relative_ci_high < -practical_threshold),
            (global_summary.p_adjusted < 0.05) & (global_summary.relative_ci_high < 0),
            (global_summary.p_adjusted < 0.05)
            & (global_summary.relative_benefit_percent > 0)
            & (global_summary.relative_ci_low <= 0),
            (global_summary.p_adjusted < 0.05)
            & (global_summary.relative_benefit_percent < 0)
            & (global_summary.relative_ci_high >= 0),
        ],
        [
            "statistically and practically better",
            "statistically better; practical threshold not established",
            "statistically and practically worse",
            "statistically worse; practical threshold not established",
            "fixed-test difference favours federation; joint bootstrap includes zero",
            "fixed-test difference favours scratch; joint bootstrap includes zero",
        ],
        default="no statistically distinct performance",
    )
    global_summary.to_csv(analysis_dir / "global_significance.csv", index=False)
    analysis_manifest = {
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "practical_threshold_percent": practical_threshold,
        "primary_metric": "rmse",
        "primary_scope": "next_fold",
        "independent_testing_units": "25 matched full-test model fits within each rolling origin",
        "bootstrap_role": "95% uncertainty intervals only; bootstrap draws are not independent observations",
        "global_claim": "equal-weight average across the four observed rolling-origin challenges, evaluated separately by endpoint and scope",
        "multiple_comparison_workflow": "repeated-measures ANOVA plus repeated-measures Tukey HSD; switch to Friedman plus Conover-Holm for strong diagnostic violations",
    }
    (analysis_dir / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n")
    print(f"Analysis written to {analysis_dir}")


if __name__ == "__main__":
    main()
