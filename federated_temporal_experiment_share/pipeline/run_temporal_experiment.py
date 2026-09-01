#!/usr/bin/env python3
"""Run a replicated rolling-origin Chemprop federation experiment.

The script deliberately predicts each trained model only once, on all future
rows.  ``analyze_temporal_experiment.py`` derives next-fold and last-fold views,
computes full-test metrics, bootstraps confidence intervals, and performs the
paired statistical comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


DEFAULT_TARGETS = ["pKi", "LogD (pH 7.40)", "Solubility (pH 7.40) (uM)"]
DEFAULT_SOLUBILITY_COLUMN = "Solubility (pH 7.40) (uM)"
LOG_SOLUBILITY_COLUMN = "log10_solubility_uM"
STRATEGIES = (
    "scratch",
    "scratch_single_task",
    "scratch_conservative_lr",
    "scratch_continual_cumulative",
    "scratch_continual_replay",
    "scratch_continual_similarity_replay",
    "scratch_continual_similarity_matched_random",
    "scratch_continual_incremental",
    "foundation_finetune",
    "foundation_finetune_single_task",
    "chemeleon_finetune",
    "chemeleon_finetune_single_task",
    "chemeleon_conservative_lr",
    "chemeleon_frozen",
    "chemeleon_staged_unfreeze",
    "chemeleon_continual_cumulative",
    "chemeleon_continual_replay",
    "chemeleon_continual_similarity_replay",
    "chemeleon_continual_similarity_matched_random",
    "chemeleon_continual_incremental",
    "foundation_conservative_lr",
    "foundation_frozen",
    "foundation_staged_unfreeze",
    "continual_cumulative",
    "continual_replay",
    "continual_similarity_replay",
    "continual_similarity_matched_random",
    "continual_incremental",
    "transductive_similarity_filtered",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--foundation", type=Path, default=Path("data/federated_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("temporal_experiment"))
    parser.add_argument("--date-column", default="DATE")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--target-columns", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--solubility-column", default=DEFAULT_SOLUBILITY_COLUMN)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--staged-head-epochs",
        type=int,
        default=10,
        help=(
            "For foundation_staged_unfreeze, train only the new head for this many epochs; "
            "the remaining --epochs are spent with the whole model unfrozen."
        ),
    )
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.20,
        help=(
            "For continual_replay, retain this fraction of every earlier temporal fold and "
            "combine it with all newly arrived rows."
        ),
    )
    parser.add_argument("--conservative-init-lr", type=float, default=1e-5)
    parser.add_argument("--conservative-max-lr", type=float, default=1e-4)
    parser.add_argument("--conservative-final-lr", type=float, default=1e-5)
    parser.add_argument("--similarity-threshold", type=float, default=0.30)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--pytorch-seeds", nargs="+", type=int, default=[1001, 1002, 1003, 1004, 1005]
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20240806)
    parser.add_argument("--practical-threshold-percent", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chemprop-command", default="chemprop")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=None,
        help=(
            "Train only these strategies and merge them into any existing experiment. "
            "By default all known strategies are checked; completed artifacts are reused."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write all data splits and CLI command logs without training or analysis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a recognisable prior output directory before starting.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip the final metric/bootstrap/significance analysis.",
    )
    return parser.parse_args()


def run(command: list[str], log_path: Path, dry_run: bool) -> None:
    printable = shlex.join(command)
    print(f"$ {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(printable + "\n")
        return
    environment = os.environ.copy()
    cache_root = log_path.parent
    environment.setdefault("MPLCONFIGDIR", str(cache_root / ".mpl-cache"))
    environment.setdefault("XDG_CACHE_HOME", str(cache_root / ".xdg-cache"))
    with log_path.open("w") as log:
        log.write(f"$ {printable}\n\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)


def optimal_contiguous_date_folds(dates: pd.Series, n_folds: int) -> np.ndarray:
    """Balance row counts while only cutting between complete ordered dates."""
    counts = dates.value_counts().sort_index()
    if len(counts) < n_folds:
        raise ValueError(f"Cannot make {n_folds} folds from only {len(counts)} unique dates")
    weights = counts.to_numpy(dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    target = cumulative[-1] / n_folds
    n_dates = len(weights)
    cost = np.full((n_folds + 1, n_dates + 1), np.inf)
    previous = np.full((n_folds + 1, n_dates + 1), -1, dtype=int)
    cost[0, 0] = 0.0
    for fold in range(1, n_folds + 1):
        for end in range(fold, n_dates + 1):
            starts = np.arange(fold - 1, end)
            segment_sizes = cumulative[end] - cumulative[starts]
            candidates = cost[fold - 1, starts] + (segment_sizes - target) ** 2
            best = int(np.argmin(candidates))
            cost[fold, end] = candidates[best]
            previous[fold, end] = starts[best]
    boundaries = [n_dates]
    end = n_dates
    for fold in range(n_folds, 0, -1):
        end = int(previous[fold, end])
        boundaries.append(end)
    boundaries.reverse()
    ordered_dates = list(counts.index)
    date_to_fold: dict[pd.Timestamp, int] = {}
    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        for date in ordered_dates[start:end]:
            date_to_fold[date] = fold
    return dates.map(date_to_fold).to_numpy(dtype=int)


def fold_summary(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    return (
        frame.groupby("temporal_fold", sort=True)
        .agg(
            n_rows=("_row_id", "size"),
            n_dates=(date_column, "nunique"),
            date_start=(date_column, "min"),
            date_end=(date_column, "max"),
        )
        .reset_index()
    )


def prepare_model_targets(
    data: pd.DataFrame, target_columns: list[str], solubility_column: str
) -> tuple[list[str], list[dict[str, str]]]:
    model_targets: list[str] = []
    transforms: list[dict[str, str]] = []
    for source_column in target_columns:
        if source_column == solubility_column:
            values = data[source_column].astype(float)
            if (values < 0).any():
                raise ValueError(f"Cannot apply log10(1 + x) to negative values in {source_column!r}")
            data[LOG_SOLUBILITY_COLUMN] = np.log10(1.0 + values)
            model_targets.append(LOG_SOLUBILITY_COLUMN)
            transforms.append(
                {
                    "source_column": source_column,
                    "model_column": LOG_SOLUBILITY_COLUMN,
                    "display_name": "Solubility",
                    "transform": "log10(1 + solubility_uM)",
                    "inverse_transform": "10**prediction - 1",
                }
            )
        else:
            model_targets.append(source_column)
            transforms.append(
                {
                    "source_column": source_column,
                    "model_column": source_column,
                    "display_name": "LogD" if source_column.startswith("LogD") else source_column,
                    "transform": "identity (already logarithmic)",
                    "inverse_transform": "identity",
                }
            )
    return model_targets, transforms


def find_trained_model(model_dir: Path) -> Path:
    best = sorted(model_dir.rglob("best.pt"))
    if len(best) == 1:
        return best[0]
    candidates = sorted(model_dir.rglob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected exactly one Chemprop checkpoint under {model_dir}; found {candidates}")


def expected_checkpoint(model_dir: Path, dry_run: bool) -> Path:
    return model_dir / "model_0" / "best.pt" if dry_run else find_trained_model(model_dir)


def train_command(
    args: argparse.Namespace,
    train_path: Path,
    model_dir: Path,
    model_targets: list[str],
    data_seed: int,
    pytorch_seed: int,
    *,
    foundation: Path | str | None = None,
    checkpoint: Path | None = None,
    freeze_encoder: bool = False,
    initialization_pass: bool = False,
    featurizer_mode: str | None = None,
    epochs_override: int | None = None,
    learning_rates: tuple[float, float, float] | None = None,
) -> list[str]:
    command = [
        args.chemprop_command,
        "train",
        "--data-path",
        str(train_path),
        "--smiles-columns",
        args.smiles_column,
        "--target-columns",
        *model_targets,
        "--output-dir",
        str(model_dir),
        "--task-type",
        "regression",
        "--split",
        "RANDOM",
        "--split-sizes",
        "0.8",
        "0.2",
        "0.0",
        "--epochs",
        "1" if initialization_pass else str(epochs_override or args.epochs),
        "--patience",
        str(args.patience),
        "--metrics",
        "rmse",
        "mae",
        "r2",
        "--show-individual-scores",
        "--data-seed",
        str(data_seed),
        "--pytorch-seed",
        str(pytorch_seed),
        "--num-workers",
        str(args.num_workers),
    ]
    if foundation is not None:
        command.extend(["--from-foundation", str(foundation)])
    if featurizer_mode is not None:
        command.extend(["--multi-hot-atom-featurizer-mode", featurizer_mode])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if freeze_encoder:
        command.append("--freeze-encoder")
    if initialization_pass:
        command.extend(
            [
                "--warmup-epochs",
                "0",
                "--init-lr",
                "1e-30",
                "--max-lr",
                "1e-30",
                "--final-lr",
                "1e-30",
            ]
        )
    elif learning_rates is not None:
        init_lr, max_lr, final_lr = learning_rates
        command.extend(
            [
                "--init-lr",
                str(init_lr),
                "--max-lr",
                str(max_lr),
                "--final-lr",
                str(final_lr),
            ]
        )
    return command


def ensure_trained(
    command: list[str], model_dir: Path, log_path: Path, dry_run: bool
) -> Path:
    if not dry_run:
        try:
            checkpoint = find_trained_model(model_dir)
            prune_lightning_checkpoints(model_dir)
            print(f"Reusing completed checkpoint: {checkpoint}", flush=True)
            return checkpoint
        except FileNotFoundError:
            if model_dir.exists() and any(model_dir.iterdir()):
                results_root = next(
                    (
                        parent
                        for parent in model_dir.parents
                        if (parent / "experiment_manifest.json").is_file()
                    ),
                    None,
                )
                if results_root is None:
                    raise RuntimeError(
                        f"Cannot safely locate the experiment root for partial model: {model_dir}"
                    )
                archive_root = results_root / "incomplete_archive"
                archive_root.mkdir(parents=True, exist_ok=True)
                archive_prefix = "__".join(model_dir.relative_to(results_root).parts)
                prior_recoveries = list(archive_root.glob(f"{archive_prefix}__*"))
                if len(prior_recoveries) >= 3:
                    raise RuntimeError(
                        "The same model has been incomplete on three separate attempts; "
                        f"this is likely a genuine training error rather than an interrupted process: {model_dir}"
                    )
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = archive_root / f"{archive_prefix}__{timestamp}"
                shutil.move(str(model_dir), str(archived))
                (archived / "RECOVERY.json").write_text(
                    json.dumps(
                        {
                            "reason": "no completed Chemprop checkpoint found",
                            "original_path": str(model_dir),
                            "archived_at": datetime.now(timezone.utc).isoformat(),
                            "recovery_attempt": len(prior_recoveries) + 1,
                            "maximum_automatic_recoveries": 3,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    f"Archived interrupted model directory and retrying: {archived}",
                    flush=True,
                )
    run(command, log_path, dry_run)
    checkpoint = expected_checkpoint(model_dir, dry_run)
    if not dry_run:
        prune_lightning_checkpoints(model_dir)
    return checkpoint


def prune_lightning_checkpoints(model_dir: Path) -> None:
    """Remove redundant trainer checkpoints after Chemprop exports ``best.pt``."""
    checkpoint_files = list(model_dir.rglob("*.ckpt"))
    if not checkpoint_files:
        return
    reclaimed = sum(path.stat().st_size for path in checkpoint_files)
    for path in checkpoint_files:
        path.unlink()
    print(
        f"Pruned {len(checkpoint_files)} redundant Lightning checkpoint(s) "
        f"({reclaimed / (1024 ** 2):.1f} MiB) from {model_dir}",
        flush=True,
    )


def ensure_prediction(
    args: argparse.Namespace,
    checkpoint: Path,
    prediction_input: Path,
    output_path: Path,
    log_path: Path,
    featurizer_mode: str | None = None,
) -> None:
    if not args.dry_run and output_path.is_file():
        prediction_complete = False
        try:
            expected_ids = pd.read_csv(prediction_input, usecols=["_row_id"])["_row_id"]
            observed_ids = pd.read_csv(output_path, usecols=["_row_id"])["_row_id"]
            prediction_complete = (
                len(observed_ids) == len(expected_ids)
                and observed_ids.is_unique
                and observed_ids.tolist() == expected_ids.tolist()
            )
        except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
            prediction_complete = False

        if prediction_complete and output_path.stat().st_mtime >= checkpoint.stat().st_mtime:
            print(f"Reusing completed predictions: {output_path}", flush=True)
            return
        if prediction_complete:
            print(f"Checkpoint is newer; refreshing stale predictions: {output_path}", flush=True)
        else:
            results_root = next(
                (
                    parent
                    for parent in output_path.parents
                    if (parent / "experiment_manifest.json").is_file()
                ),
                None,
            )
            if results_root is None:
                raise RuntimeError(
                    f"Cannot safely locate the experiment root for partial prediction: {output_path}"
                )
            archive_root = results_root / "incomplete_archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            archive_prefix = "__".join(output_path.relative_to(results_root).parts)
            prior_recoveries = list(archive_root.glob(f"{archive_prefix}__*"))
            if len(prior_recoveries) >= 3:
                raise RuntimeError(
                    "The same prediction has been incomplete on three separate attempts; "
                    f"this is likely a genuine prediction error: {output_path}"
                )
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archived = archive_root / f"{archive_prefix}__{timestamp}"
            shutil.move(str(output_path), str(archived))
            print(f"Archived interrupted prediction and retrying: {archived}", flush=True)
    command = [
        args.chemprop_command,
        "predict",
        "--test-path",
        str(prediction_input),
        "--smiles-columns",
        args.smiles_column,
        "--model-paths",
        str(checkpoint),
        "--output",
        str(output_path),
        "--num-workers",
        str(args.num_workers),
    ]
    if featurizer_mode is not None:
        command.extend(["--multi-hot-atom-featurizer-mode", featurizer_mode])
    run(command, log_path, args.dry_run)


def prepare_replay_training_file(
    data: pd.DataFrame,
    origin: int,
    data_seed: int,
    replay_fraction: float,
    smiles_column: str,
    model_targets: list[str],
    output_path: Path,
) -> dict[str, int | float | str]:
    """Combine the complete new fold with a fold-stratified historical replay sample."""
    new_rows = data.loc[data.temporal_fold == origin]
    history = data.loc[data.temporal_fold < origin]
    rng = np.random.default_rng(data_seed + origin * 100_003)
    sampled_indices: list[int] = []
    for _, fold_rows in history.groupby("temporal_fold", sort=True):
        n_sample = max(1, int(round(len(fold_rows) * replay_fraction)))
        n_sample = min(n_sample, len(fold_rows))
        sampled_indices.extend(rng.choice(fold_rows.index.to_numpy(), n_sample, replace=False))
    replay_rows = history.loc[sorted(sampled_indices)]
    combined = (
        pd.concat([replay_rows, new_rows], ignore_index=False)
        .sort_values(["temporal_fold", "_row_id"], kind="stable")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined[[smiles_column, *model_targets]].to_csv(output_path, index=False)
    return {
        "origin": origin,
        "data_seed": data_seed,
        "replay_fraction": replay_fraction,
        "n_new_rows": len(new_rows),
        "n_replay_rows": len(replay_rows),
        "n_training_rows": len(combined),
        "path": str(output_path),
    }


def compute_fingerprints(
    data: pd.DataFrame, smiles_column: str, radius: int, n_bits: int
) -> dict[int, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints: dict[int, object] = {}
    for index, smiles in data[smiles_column].items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse SMILES at row index {index}: {smiles}")
        fingerprints[index] = generator.GetFingerprint(molecule)
    return fingerprints


def maximum_similarities(
    candidate_indices: list[int], reference_indices: list[int], fingerprints: dict[int, object]
) -> pd.Series:
    if not reference_indices:
        raise ValueError("Similarity filtering requires at least one reference molecule")
    references = [fingerprints[index] for index in reference_indices]
    values = {
        index: max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], references))
        for index in candidate_indices
    }
    return pd.Series(values, dtype=float)


def prepare_similarity_training_files(
    data: pd.DataFrame,
    data_seeds: list[int],
    n_folds: int,
    threshold: float,
    smiles_column: str,
    model_targets: list[str],
    output_dir: Path,
    fingerprints: dict[int, object],
) -> tuple[
    dict[int, Path],
    dict[tuple[int, int], Path],
    dict[int, Path],
    list[dict[str, int | float | str]],
]:
    """Prepare prospective, matched-random, and transductive similarity subsets."""
    prospective_paths: dict[int, Path] = {}
    matched_paths: dict[tuple[int, int], Path] = {}
    transductive_paths: dict[int, Path] = {}
    records: list[dict[str, int | float | str]] = []

    for origin in range(1, n_folds):
        origin_dir = output_dir / f"origin_{origin:02d}" / "training"
        cumulative = data.loc[data.temporal_fold <= origin]
        next_fold = data.loc[data.temporal_fold == origin + 1]
        trans_similarity = maximum_similarities(
            cumulative.index.to_list(), next_fold.index.to_list(), fingerprints
        )
        trans_selected = cumulative.loc[trans_similarity[trans_similarity >= threshold].index]
        if trans_selected.empty:
            raise ValueError(
                f"No training molecules pass similarity threshold {threshold} at origin {origin}"
            )
        trans_path = origin_dir / "transductive_similarity_filtered.csv"
        trans_selected[[smiles_column, *model_targets]].to_csv(trans_path, index=False)
        transductive_paths[origin] = trans_path
        records.append(
            {
                "strategy": "transductive_similarity_filtered",
                "origin": origin,
                "data_seed": "all",
                "reference_fold": origin + 1,
                "n_candidates": len(cumulative),
                "n_selected": len(trans_selected),
                "threshold": threshold,
                "mean_max_similarity": float(trans_similarity.mean()),
                "path": str(trans_path),
            }
        )

        if origin == 1:
            continue
        new_rows = data.loc[data.temporal_fold == origin]
        history = data.loc[data.temporal_fold < origin]
        historical_similarity = maximum_similarities(
            history.index.to_list(), new_rows.index.to_list(), fingerprints
        )
        similar_history = history.loc[historical_similarity[historical_similarity >= threshold].index]
        prospective = pd.concat([similar_history, new_rows]).sort_values(
            ["temporal_fold", "_row_id"], kind="stable"
        )
        prospective_path = origin_dir / "continual_similarity_replay.csv"
        prospective[[smiles_column, *model_targets]].to_csv(prospective_path, index=False)
        prospective_paths[origin] = prospective_path
        records.append(
            {
                "strategy": "continual_similarity_replay",
                "origin": origin,
                "data_seed": "all",
                "reference_fold": origin,
                "n_candidates": len(history),
                "n_selected": len(similar_history),
                "threshold": threshold,
                "mean_max_similarity": float(historical_similarity.mean()),
                "path": str(prospective_path),
            }
        )

        selected_counts = similar_history.groupby("temporal_fold").size().to_dict()
        for data_seed in data_seeds:
            rng = np.random.default_rng(data_seed + origin * 200_003)
            random_indices: list[int] = []
            for historical_fold, fold_rows in history.groupby("temporal_fold", sort=True):
                n_selected = int(selected_counts.get(historical_fold, 0))
                if n_selected:
                    random_indices.extend(
                        rng.choice(fold_rows.index.to_numpy(), n_selected, replace=False)
                    )
            random_history = history.loc[sorted(random_indices)]
            matched = pd.concat([random_history, new_rows]).sort_values(
                ["temporal_fold", "_row_id"], kind="stable"
            )
            matched_path = origin_dir / f"similarity_matched_random_data_seed_{data_seed}.csv"
            matched[[smiles_column, *model_targets]].to_csv(matched_path, index=False)
            matched_paths[(origin, data_seed)] = matched_path
            records.append(
                {
                    "strategy": "continual_similarity_matched_random",
                    "origin": origin,
                    "data_seed": data_seed,
                    "reference_fold": origin,
                    "n_candidates": len(history),
                    "n_selected": len(random_history),
                    "threshold": threshold,
                    "mean_max_similarity": float("nan"),
                    "path": str(matched_path),
                }
            )
    return prospective_paths, matched_paths, transductive_paths, records


def target_slug(target: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in target)
    return "_".join(part for part in cleaned.split("_") if part)


def combine_single_task_predictions(
    prediction_paths: dict[str, Path], output_path: Path
) -> None:
    combined: pd.DataFrame | None = None
    for target, path in prediction_paths.items():
        frame = pd.read_csv(path)
        required = ["_row_id", target]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Single-task prediction {path} is missing {missing}")
        selected = frame[required]
        combined = (
            selected
            if combined is None
            else combined.merge(selected, on="_row_id", how="inner", validate="one_to_one")
        )
    if combined is None:
        raise ValueError("No single-task predictions were supplied")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


def prepare_output(args: argparse.Namespace, output_dir: Path) -> None:
    marker = output_dir / "experiment_manifest.json"
    if args.overwrite and output_dir.exists():
        if any(output_dir.iterdir()) and not marker.is_file():
            raise ValueError(f"Refusing to overwrite unrecognised directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_reusable_experiment(
    existing: dict[str, object], args: argparse.Namespace, data_path: Path, foundation_path: Path
) -> None:
    """Prevent accidental reuse when a model-defining setting has changed."""
    expected = {
        "data": str(data_path),
        "foundation": str(foundation_path),
        "n_folds": args.n_folds,
        "source_targets": list(args.target_columns),
        "epochs": args.epochs,
        "patience": args.patience,
        "chemprop_split_sizes": [0.8, 0.2, 0.0],
        "data_seeds": args.data_seeds,
        "pytorch_seeds": args.pytorch_seeds,
    }
    mismatches = {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    existing_staged = existing.get("staged_unfreezing")
    if isinstance(existing_staged, dict):
        requested_staged = {
            "head_only_epochs": args.staged_head_epochs,
            "fully_unfrozen_epochs": args.epochs - args.staged_head_epochs,
        }
        for key, value in requested_staged.items():
            if existing_staged.get(key) != value:
                mismatches[f"staged_unfreezing.{key}"] = {
                    "existing": existing_staged.get(key),
                    "requested": value,
                }
    existing_replay = existing.get("historical_replay")
    if isinstance(existing_replay, dict) and existing_replay.get("fraction") != args.replay_fraction:
        mismatches["historical_replay.fraction"] = {
            "existing": existing_replay.get("fraction"),
            "requested": args.replay_fraction,
        }
    existing_conservative = existing.get("conservative_learning_rate")
    requested_rates = {
        "init_lr": args.conservative_init_lr,
        "max_lr": args.conservative_max_lr,
        "final_lr": args.conservative_final_lr,
    }
    if isinstance(existing_conservative, dict):
        for key, value in requested_rates.items():
            if existing_conservative.get(key) != value:
                mismatches[f"conservative_learning_rate.{key}"] = {
                    "existing": existing_conservative.get(key),
                    "requested": value,
                }
    existing_similarity = existing.get("similarity_filtering")
    requested_similarity = {
        "threshold": args.similarity_threshold,
        "radius": args.fingerprint_radius,
        "n_bits": args.fingerprint_bits,
    }
    if isinstance(existing_similarity, dict):
        for key, value in requested_similarity.items():
            if existing_similarity.get(key) != value:
                mismatches[f"similarity_filtering.{key}"] = {
                    "existing": existing_similarity.get(key),
                    "requested": value,
                }
    if mismatches:
        details = json.dumps(mismatches, indent=2)
        raise ValueError(
            "Existing checkpoints cannot safely be reused because model-defining settings differ:\n"
            f"{details}\nUse a new --output-dir or intentionally restart with --overwrite."
        )


def main() -> None:
    args = parse_args()
    if args.n_folds != 5:
        raise ValueError("This inferential design expects exactly five temporal folds")
    if len(args.data_seeds) != 5 or len(args.pytorch_seeds) != 5:
        raise ValueError("The planned crossed design requires exactly five data and five PyTorch seeds")
    if len(set(args.data_seeds)) != 5 or len(set(args.pytorch_seeds)) != 5:
        raise ValueError("Seeds must be unique within each seed family")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if not 1 <= args.staged_head_epochs < args.epochs:
        raise ValueError("--staged-head-epochs must be at least 1 and smaller than --epochs")
    if not 0 < args.replay_fraction <= 1:
        raise ValueError("--replay-fraction must be greater than 0 and no greater than 1")
    if not (
        0 < args.conservative_init_lr
        <= args.conservative_max_lr
        and 0 < args.conservative_final_lr <= args.conservative_max_lr
    ):
        raise ValueError("Conservative learning rates must be positive and bounded by max LR")
    if not 0 <= args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold must be between 0 and 1")
    if args.fingerprint_radius < 1 or args.fingerprint_bits < 64:
        raise ValueError("Fingerprint radius must be positive and fingerprint bits at least 64")

    data_path = args.data.resolve()
    foundation_path = args.foundation.resolve()
    output_dir = args.output_dir.resolve()
    if not data_path.is_file() or not foundation_path.is_file():
        raise FileNotFoundError(f"Expected {data_path} and {foundation_path}")
    if not args.dry_run and shutil.which(args.chemprop_command) is None:
        raise RuntimeError("Run this script inside the 'chemprop' conda environment")
    existing_manifest_path = output_dir / "experiment_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text())
        if existing_manifest_path.is_file() and not args.overwrite
        else {}
    )
    if existing_manifest:
        validate_reusable_experiment(existing_manifest, args, data_path, foundation_path)
    prepare_output(args, output_dir)

    requested_strategies = list(args.strategies or STRATEGIES)
    existing_strategies = list(existing_manifest.get("strategies", []))
    if "scratch" not in existing_strategies and "scratch" not in requested_strategies:
        requested_strategies.insert(0, "scratch")
    if (
        any(
            strategy in requested_strategies
            for strategy in (
                "foundation_finetune_single_task",
                "chemeleon_finetune_single_task",
            )
        )
        and "scratch_single_task" not in existing_strategies
        and "scratch_single_task" not in requested_strategies
    ):
        requested_strategies.append("scratch_single_task")
    if any(strategy.startswith("continual_") for strategy in requested_strategies):
        if "foundation_finetune" not in requested_strategies:
            requested_strategies.append("foundation_finetune")
    if any(strategy.startswith("scratch_continual_") for strategy in requested_strategies):
        if "scratch" not in requested_strategies:
            requested_strategies.append("scratch")
    if any(strategy.startswith("chemeleon_continual_") for strategy in requested_strategies):
        if "chemeleon_finetune" not in requested_strategies:
            requested_strategies.append("chemeleon_finetune")
    requested_strategies = [strategy for strategy in STRATEGIES if strategy in requested_strategies]
    analysis_strategies = [
        strategy
        for strategy in STRATEGIES
        if strategy in set(existing_strategies) | set(requested_strategies)
    ]
    matched_comparator_overrides = {
        "foundation_finetune_single_task": "scratch_single_task",
        "chemeleon_finetune_single_task": "scratch_single_task",
        "foundation_conservative_lr": "scratch_conservative_lr",
        "chemeleon_conservative_lr": "scratch_conservative_lr",
        "continual_cumulative": "scratch_continual_cumulative",
        "chemeleon_continual_cumulative": "scratch_continual_cumulative",
        "continual_replay": "scratch_continual_replay",
        "chemeleon_continual_replay": "scratch_continual_replay",
        "continual_similarity_replay": "scratch_continual_similarity_replay",
        "chemeleon_continual_similarity_replay": "scratch_continual_similarity_replay",
        "continual_similarity_matched_random": "scratch_continual_similarity_matched_random",
        "chemeleon_continual_similarity_matched_random": "scratch_continual_similarity_matched_random",
        "continual_incremental": "scratch_continual_incremental",
        "chemeleon_continual_incremental": "scratch_continual_incremental",
    }

    data = pd.read_csv(data_path)
    required = [args.date_column, args.smiles_column, *args.target_columns]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("DATE, SMILES, and all targets must be present")
    parsed_dates = pd.to_datetime(data[args.date_column], errors="raise")
    data = data.copy()
    model_targets, target_transforms = prepare_model_targets(
        data, list(args.target_columns), args.solubility_column
    )
    data[args.date_column] = parsed_dates.dt.strftime("%Y-%m-%d")
    data.insert(0, "_row_id", np.arange(len(data), dtype=int))
    data["temporal_fold"] = optimal_contiguous_date_folds(parsed_dates, args.n_folds)
    data = data.sort_values([args.date_column, "_row_id"], kind="stable").reset_index(drop=True)
    data.to_csv(output_dir / "fold_assignments.csv", index=False)
    summary = fold_summary(data, args.date_column)
    summary.to_csv(output_dir / "fold_summary.csv", index=False)

    manifest = {
        "schema_version": 2,
        "data": str(data_path),
        "foundation": str(foundation_path),
        "n_folds": args.n_folds,
        "rolling_origins": list(range(1, args.n_folds)),
        "fold_method": "optimal contiguous complete-date buckets balanced by row count; dates never cross buckets",
        "strategies": analysis_strategies,
        "requested_strategies_this_run": requested_strategies,
        "resume_policy": "completed best.pt checkpoints and all-future prediction CSVs are reused; only missing requested artifacts are generated",
        "strategy_descriptions": {
            "scratch": "independent random initialization; cumulative data",
            "scratch_single_task": "three independent randomly initialized models, one per endpoint; cumulative data",
            "scratch_conservative_lr": "independent random initialization with the same conservative learning-rate schedule used for pretrained encoders; cumulative data",
            "scratch_continual_cumulative": "origin 1 scratch model; later origins checkpoint the previous whole model and retrain on cumulative data",
            "scratch_continual_replay": "origin 1 scratch model; later origins checkpoint the previous model and train on all new rows plus fold-stratified historical replay",
            "scratch_continual_similarity_replay": "origin 1 scratch model; later origins checkpoint the previous model and train on all new rows plus history similar to the latest observed fold",
            "scratch_continual_similarity_matched_random": "size-matched random control for scratch continual similarity replay",
            "scratch_continual_incremental": "origin 1 scratch model; later origins checkpoint the previous whole model and train only on the newly arrived fold",
            "foundation_finetune": "foundation encoder plus newly initialized task head at every origin; cumulative data",
            "foundation_finetune_single_task": "three independent models initialized from the supplied foundation encoder, one new single-endpoint head per model and origin; cumulative data",
            "chemeleon_finetune": "Chemprop CheMeleon foundation encoder plus newly initialized task head at every origin; cumulative data; V2 atom featurizer",
            "chemeleon_finetune_single_task": "three independent CheMeleon-initialized models, one new single-endpoint head per model and origin; cumulative data; V2 atom featurizer",
            "chemeleon_conservative_lr": "CheMeleon encoder plus a new head at every origin with the same conservative learning-rate schedule; cumulative data; V2 atom featurizer",
            "chemeleon_frozen": "CheMeleon encoder frozen after an inert task-head initialization; only the new task head learns; V2 atom featurizer",
            "chemeleon_staged_unfreeze": "CheMeleon encoder frozen for head warm-up and then fully unfrozen, with a compute-matched epoch budget; V2 atom featurizer",
            "chemeleon_continual_cumulative": "origin 1 CheMeleon fine-tune; later origins checkpoint the previous whole model and retrain on cumulative data",
            "chemeleon_continual_replay": "origin 1 CheMeleon fine-tune; later origins checkpoint the previous model and train on all new rows plus historical replay",
            "chemeleon_continual_similarity_replay": "origin 1 CheMeleon fine-tune; later origins checkpoint the previous model and use similarity replay",
            "chemeleon_continual_similarity_matched_random": "size-matched random control for CheMeleon continual similarity replay",
            "chemeleon_continual_incremental": "origin 1 CheMeleon fine-tune; later origins checkpoint the previous whole model and train only on the newly arrived fold",
            "foundation_conservative_lr": "foundation encoder plus newly initialized task head at every origin; entire model unfrozen with a 10x lower default learning-rate schedule; cumulative data",
            "foundation_frozen": "foundation encoder frozen exactly; only a newly initialized task head learns; cumulative data",
            "foundation_staged_unfreeze": "new head trains against an exactly frozen foundation encoder, then the full model is unfrozen; cumulative data and a compute-matched total epoch budget",
            "continual_cumulative": "origin 1 foundation fine-tune; later origins checkpoint previous whole model and retrain on cumulative data",
            "continual_replay": "origin 1 foundation fine-tune; later origins checkpoint the previous whole model and train on all new rows plus a fold-stratified sample of historical rows",
            "continual_similarity_replay": "origin 1 foundation fine-tune; later origins checkpoint the previous model and train on all new rows plus historical rows similar to the latest observed fold",
            "continual_similarity_matched_random": "size-matched random control for continual_similarity_replay, stratified by historical fold and varied by data seed",
            "continual_incremental": "origin 1 foundation fine-tune; later origins checkpoint previous whole model and train only on the newly arrived fold",
            "transductive_similarity_filtered": "new foundation head at each origin, trained on cumulative rows similar to the unlabeled next-fold query structures; excluded from primary prospective claims",
        },
        "origin_1_aliases": {
            "scratch_continual_cumulative": "scratch",
            "scratch_continual_replay": "scratch",
            "scratch_continual_similarity_replay": "scratch",
            "scratch_continual_similarity_matched_random": "scratch",
            "scratch_continual_incremental": "scratch",
            "chemeleon_continual_cumulative": "chemeleon_finetune",
            "chemeleon_continual_replay": "chemeleon_finetune",
            "chemeleon_continual_similarity_replay": "chemeleon_finetune",
            "chemeleon_continual_similarity_matched_random": "chemeleon_finetune",
            "chemeleon_continual_incremental": "chemeleon_finetune",
            "continual_cumulative": "foundation_finetune",
            "continual_replay": "foundation_finetune",
            "continual_similarity_replay": "foundation_finetune",
            "continual_similarity_matched_random": "foundation_finetune",
            "continual_incremental": "foundation_finetune",
        },
        "remote_foundation_models": {
            "chemeleon_finetune": "CheMeleon is resolved by Chemprop via --from-foundation CheMeleon and downloaded to Chemprop's cache on first use"
        },
        "evaluation_scopes": {
            "next_fold": "fold immediately after the rolling origin (primary)",
            "all_future": "all rows after the rolling origin (secondary; overlaps across origins)",
            "last_fold": "fixed fifth fold (secondary; overlaps across origins)",
        },
        "source_targets": list(args.target_columns),
        "model_targets": model_targets,
        "target_transforms": target_transforms,
        "chemprop_target_scaling": "Chemprop applies training-only target standardization and reverses it for predictions.",
        "epochs": args.epochs,
        "patience": args.patience,
        "chemprop_split_sizes": [0.8, 0.2, 0.0],
        "data_seeds": args.data_seeds,
        "pytorch_seeds": args.pytorch_seeds,
        "replicate_design": "5 data-seed splits crossed with 5 PyTorch seeds = 25 matched model fits per origin and strategy",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_design": "full-size paired row resampling plus crossed data-seed/PyTorch-seed resampling; draws are uncertainty samples, not independent observations",
        "practical_threshold_percent": args.practical_threshold_percent,
        "prediction_design": "one all-future prediction file per trained model; next and last scopes are derived by _row_id",
        "frozen_encoder_initialization": "a numerically inert one-epoch CLI pass from foundation creates a task-specific head and scaler; a separate checkpoint preparation process restores the foundation message-passing/aggregation tensors bit-for-bit and resets batch normalization; the 100-epoch CLI fit uses --checkpoint and --freeze-encoder",
        "chemeleon_frozen_initialization": "a numerically inert one-epoch CLI pass from CheMeleon creates the task-specific head and scaler without changing encoder weights; subsequent frozen or staged phases use --checkpoint with V2 atom featurization",
        "fairness_strategy_grid": {
            "supplied_federation": "standard MTL, single-task, conservative LR, frozen, staged unfreeze, and cumulative/replay/similarity/incremental continual variants",
            "chemeleon": "the same standard MTL, single-task, conservative LR, frozen, staged unfreeze, and cumulative/replay/similarity/incremental continual variants",
            "scratch": "standard MTL, single-task, conservative LR, and the same cumulative/replay/similarity/incremental continual variants; frozen and staged unfreezing are structurally inapplicable without a pretrained trunk",
            "hyperparameter_optimization": "none for any initialization family",
        },
        "staged_unfreezing": {
            "head_only_epochs": args.staged_head_epochs,
            "fully_unfrozen_epochs": args.epochs - args.staged_head_epochs,
            "total_counted_training_epochs": args.epochs,
            "phase_1": "--checkpoint with --freeze-encoder",
            "phase_2": "--checkpoint without --freeze-encoder",
        },
        "historical_replay": {
            "fraction": args.replay_fraction,
            "sampling": "without replacement, stratified by every earlier temporal fold, varied by data seed",
            "new_data": "all rows in the newly arrived fold",
        },
        "conservative_learning_rate": {
            "init_lr": args.conservative_init_lr,
            "max_lr": args.conservative_max_lr,
            "final_lr": args.conservative_final_lr,
            "comparison": "10x below Chemprop defaults for all three schedule points",
        },
        "similarity_filtering": {
            "threshold": args.similarity_threshold,
            "fingerprint": "ECFP4 / Morgan radius 2 by default",
            "radius": args.fingerprint_radius,
            "n_bits": args.fingerprint_bits,
            "metric": "maximum Tanimoto similarity to any reference molecule",
            "prospective_reference": "latest fully observed training fold",
            "transductive_reference": "unlabeled next-fold query structures",
        },
        "strategy_families": {
            "north_star_prospective": [
                strategy
                for strategy in analysis_strategies
                if strategy != "transductive_similarity_filtered"
            ],
            "prospective_multitask": [
                strategy
                for strategy in analysis_strategies
                if "single_task" not in strategy
                and strategy != "transductive_similarity_filtered"
            ],
            "task_formulation": [
                strategy
                for strategy in (
                    "scratch",
                    "scratch_single_task",
                    "foundation_finetune",
                    "foundation_finetune_single_task",
                    "chemeleon_finetune",
                    "chemeleon_finetune_single_task",
                )
                if strategy in analysis_strategies
            ],
            "transductive_multitask": [
                strategy
                for strategy in (
                    "scratch",
                    "foundation_finetune",
                    "transductive_similarity_filtered",
                )
                if strategy in analysis_strategies
            ],
        },
        "strategy_comparators": {
            strategy: (
                matched_comparator_overrides[strategy]
                if matched_comparator_overrides.get(strategy) in analysis_strategies
                else "scratch"
            )
            for strategy in analysis_strategies
            if strategy not in {"scratch"}
        },
        "strategy_primary_family": {
            strategy: (
                "task_formulation"
                if "single_task" in strategy
                else "transductive_multitask"
                if strategy == "transductive_similarity_filtered"
                else "prospective_multitask"
            )
            for strategy in analysis_strategies
            if strategy != "scratch"
        },
        "folds": summary.to_dict(orient="records"),
    }
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    prepared: dict[int, dict[str, Path]] = {}
    for origin in range(1, args.n_folds):
        origin_dir = output_dir / f"origin_{origin:02d}"
        cumulative = data.loc[data.temporal_fold <= origin]
        incremental = data.loc[data.temporal_fold == origin]
        future = data.loc[data.temporal_fold > origin]
        train_dir = origin_dir / "training"
        eval_dir = origin_dir / "evaluation"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        cumulative_path = train_dir / "cumulative.csv"
        incremental_path = train_dir / "incremental.csv"
        truth_path = eval_dir / "all_future_truth.csv"
        input_path = eval_dir / "all_future_prediction_input.csv"
        cumulative[[args.smiles_column, *model_targets]].to_csv(cumulative_path, index=False)
        incremental[[args.smiles_column, *model_targets]].to_csv(incremental_path, index=False)
        future.to_csv(truth_path, index=False)
        future[["_row_id", args.smiles_column]].to_csv(input_path, index=False)
        prepared[origin] = {
            "cumulative": cumulative_path,
            "incremental": incremental_path,
            "truth": truth_path,
            "prediction_input": input_path,
        }

    replay_paths: dict[tuple[int, int], Path] = {}
    replay_records: list[dict[str, int | float | str]] = []
    if any(
        strategy in requested_strategies
        for strategy in (
            "continual_replay",
            "scratch_continual_replay",
            "chemeleon_continual_replay",
        )
    ):
        for origin in range(2, args.n_folds):
            for data_seed in args.data_seeds:
                replay_path = (
                    output_dir
                    / f"origin_{origin:02d}"
                    / "training"
                    / f"replay_data_seed_{data_seed}.csv"
                )
                replay_paths[(origin, data_seed)] = replay_path
                replay_records.append(
                    prepare_replay_training_file(
                        data,
                        origin,
                        data_seed,
                        args.replay_fraction,
                        args.smiles_column,
                        model_targets,
                        replay_path,
                    )
                )
        pd.DataFrame(replay_records).to_csv(output_dir / "historical_replay_summary.csv", index=False)

    similarity_prospective_paths: dict[int, Path] = {}
    similarity_matched_paths: dict[tuple[int, int], Path] = {}
    transductive_similarity_paths: dict[int, Path] = {}
    similarity_strategies = {
        "continual_similarity_replay",
        "continual_similarity_matched_random",
        "scratch_continual_similarity_replay",
        "scratch_continual_similarity_matched_random",
        "chemeleon_continual_similarity_replay",
        "chemeleon_continual_similarity_matched_random",
        "transductive_similarity_filtered",
    }
    if similarity_strategies.intersection(requested_strategies):
        fingerprints = compute_fingerprints(
            data, args.smiles_column, args.fingerprint_radius, args.fingerprint_bits
        )
        (
            similarity_prospective_paths,
            similarity_matched_paths,
            transductive_similarity_paths,
            similarity_records,
        ) = prepare_similarity_training_files(
            data,
            args.data_seeds,
            args.n_folds,
            args.similarity_threshold,
            args.smiles_column,
            model_targets,
            output_dir,
            fingerprints,
        )
        pd.DataFrame(similarity_records).to_csv(
            output_dir / "similarity_filtering_summary.csv", index=False
        )

    for data_seed in args.data_seeds:
        for pytorch_seed in args.pytorch_seeds:
            replicate_id = f"data_{data_seed}__torch_{pytorch_seed}"
            previous_continual: dict[str, Path] = {}
            for origin in range(1, args.n_folds):
                origin_dir = output_dir / f"origin_{origin:02d}"
                replicate_dir = origin_dir / "replicates" / replicate_id
                models_dir = replicate_dir / "models"
                predictions_dir = replicate_dir / "predictions"
                logs_dir = replicate_dir / "logs"
                predictions_dir.mkdir(parents=True, exist_ok=True)
                paths = prepared[origin]

                checkpoints: dict[str, Path] = {}
                if "scratch" in requested_strategies:
                    scratch_dir = models_dir / "scratch"
                    checkpoints["scratch"] = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            scratch_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                        ),
                        scratch_dir,
                        logs_dir / "train_scratch.log",
                        args.dry_run,
                    )

                for single_task_strategy, single_task_foundation, single_task_featurizer in (
                    ("scratch_single_task", None, None),
                    ("foundation_finetune_single_task", foundation_path, None),
                    ("chemeleon_finetune_single_task", "CheMeleon", "V2"),
                ):
                    if single_task_strategy not in requested_strategies:
                        continue
                    component_predictions: dict[str, Path] = {}
                    for target in model_targets:
                        slug = target_slug(target)
                        component_model_dir = models_dir / single_task_strategy / slug
                        component_checkpoint = ensure_trained(
                            train_command(
                                args,
                                paths["cumulative"],
                                component_model_dir,
                                [target],
                                data_seed,
                                pytorch_seed,
                                foundation=single_task_foundation,
                                featurizer_mode=single_task_featurizer,
                            ),
                            component_model_dir,
                            logs_dir / f"train_{single_task_strategy}_{slug}.log",
                            args.dry_run,
                        )
                        component_prediction = (
                            predictions_dir / f"{single_task_strategy}_{slug}_all_future.csv"
                        )
                        ensure_prediction(
                            args,
                            component_checkpoint,
                            paths["prediction_input"],
                            component_prediction,
                            logs_dir / f"predict_{single_task_strategy}_{slug}.log",
                            featurizer_mode=single_task_featurizer,
                        )
                        component_predictions[target] = component_prediction
                    combined_prediction = (
                        predictions_dir / f"{single_task_strategy}_all_future.csv"
                    )
                    if args.dry_run:
                        combined_prediction.write_text(
                            "COMBINE "
                            + " ".join(str(path) for path in component_predictions.values())
                            + "\n"
                        )
                    else:
                        combine_single_task_predictions(component_predictions, combined_prediction)

                if "foundation_finetune" in requested_strategies:
                    foundation_dir = models_dir / "foundation_finetune"
                    checkpoints["foundation_finetune"] = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            foundation_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation=foundation_path,
                        ),
                        foundation_dir,
                        logs_dir / "train_foundation_finetune.log",
                        args.dry_run,
                    )

                if "chemeleon_finetune" in requested_strategies:
                    chemeleon_dir = models_dir / "chemeleon_finetune"
                    checkpoints["chemeleon_finetune"] = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            chemeleon_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation="CheMeleon",
                            featurizer_mode="V2",
                        ),
                        chemeleon_dir,
                        logs_dir / "train_chemeleon_finetune.log",
                        args.dry_run,
                    )

                for conservative_strategy, conservative_foundation, conservative_featurizer in (
                    ("scratch_conservative_lr", None, None),
                    ("foundation_conservative_lr", foundation_path, None),
                    ("chemeleon_conservative_lr", "CheMeleon", "V2"),
                ):
                    if conservative_strategy not in requested_strategies:
                        continue
                    conservative_dir = models_dir / conservative_strategy
                    checkpoints[conservative_strategy] = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            conservative_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation=conservative_foundation,
                            featurizer_mode=conservative_featurizer,
                            learning_rates=(
                                args.conservative_init_lr,
                                args.conservative_max_lr,
                                args.conservative_final_lr,
                            ),
                        ),
                        conservative_dir,
                        logs_dir / f"train_{conservative_strategy}.log",
                        args.dry_run,
                    )

                if "transductive_similarity_filtered" in requested_strategies:
                    transductive_dir = models_dir / "transductive_similarity_filtered"
                    checkpoints["transductive_similarity_filtered"] = ensure_trained(
                        train_command(
                            args,
                            transductive_similarity_paths[origin],
                            transductive_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation=foundation_path,
                        ),
                        transductive_dir,
                        logs_dir / "train_transductive_similarity_filtered.log",
                        args.dry_run,
                    )

                if "foundation_frozen" in requested_strategies:
                    frozen_init_dir = models_dir / "foundation_frozen_initializer"
                    frozen_init = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            frozen_init_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation=foundation_path,
                            initialization_pass=True,
                        ),
                        frozen_init_dir,
                        logs_dir / "initialize_foundation_frozen.log",
                        args.dry_run,
                    )
                    restored_frozen_init = frozen_init_dir / "foundation_encoder_restored.pt"
                    if args.dry_run or not restored_frozen_init.is_file():
                        restore_command = [
                            sys.executable,
                            str(Path(__file__).with_name("restore_foundation_encoder.py").resolve()),
                            "--task-checkpoint",
                            str(frozen_init),
                            "--foundation",
                            str(foundation_path),
                            "--output",
                            str(restored_frozen_init),
                        ]
                        run(
                            restore_command,
                            logs_dir / "restore_foundation_encoder.log",
                            args.dry_run,
                        )
                    frozen_dir = models_dir / "foundation_frozen"
                    checkpoints["foundation_frozen"] = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            frozen_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            checkpoint=restored_frozen_init,
                            freeze_encoder=True,
                        ),
                        frozen_dir,
                        logs_dir / "train_foundation_frozen.log",
                        args.dry_run,
                    )

                if "foundation_staged_unfreeze" in requested_strategies:
                    staged_init_dir = models_dir / "foundation_frozen_initializer"
                    staged_init = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            staged_init_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation=foundation_path,
                            initialization_pass=True,
                        ),
                        staged_init_dir,
                        logs_dir / "initialize_foundation_staged_unfreeze.log",
                        args.dry_run,
                    )
                    restored_staged_init = staged_init_dir / "foundation_encoder_restored.pt"
                    if args.dry_run or not restored_staged_init.is_file():
                        restore_command = [
                            sys.executable,
                            str(Path(__file__).with_name("restore_foundation_encoder.py").resolve()),
                            "--task-checkpoint",
                            str(staged_init),
                            "--foundation",
                            str(foundation_path),
                            "--output",
                            str(restored_staged_init),
                        ]
                        run(
                            restore_command,
                            logs_dir / "restore_foundation_staged_unfreeze.log",
                            args.dry_run,
                        )

                    head_warmup_dir = models_dir / "foundation_staged_unfreeze_head_warmup"
                    head_warmup = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            head_warmup_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            checkpoint=restored_staged_init,
                            freeze_encoder=True,
                            epochs_override=args.staged_head_epochs,
                        ),
                        head_warmup_dir,
                        logs_dir / "train_foundation_staged_unfreeze_head_warmup.log",
                        args.dry_run,
                    )
                    staged_dir = models_dir / "foundation_staged_unfreeze"
                    checkpoints["foundation_staged_unfreeze"] = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            staged_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            checkpoint=head_warmup,
                            epochs_override=args.epochs - args.staged_head_epochs,
                        ),
                        staged_dir,
                        logs_dir / "train_foundation_staged_unfreeze_full_model.log",
                        args.dry_run,
                    )

                if {
                    "chemeleon_frozen",
                    "chemeleon_staged_unfreeze",
                }.intersection(requested_strategies):
                    chemeleon_init_dir = models_dir / "chemeleon_frozen_initializer"
                    chemeleon_init = ensure_trained(
                        train_command(
                            args,
                            paths["cumulative"],
                            chemeleon_init_dir,
                            model_targets,
                            data_seed,
                            pytorch_seed,
                            foundation="CheMeleon",
                            initialization_pass=True,
                            featurizer_mode="V2",
                        ),
                        chemeleon_init_dir,
                        logs_dir / "initialize_chemeleon_frozen.log",
                        args.dry_run,
                    )
                    restored_chemeleon_init = (
                        chemeleon_init_dir / "chemeleon_encoder_restored.pt"
                    )
                    if args.dry_run or not restored_chemeleon_init.is_file():
                        chemeleon_cache = Path.home() / ".chemprop" / "chemeleon_mp.pt"
                        restore_command = [
                            sys.executable,
                            str(Path(__file__).with_name("restore_foundation_encoder.py").resolve()),
                            "--task-checkpoint",
                            str(chemeleon_init),
                            "--foundation",
                            str(chemeleon_cache),
                            "--chemeleon-message-passing",
                            "--output",
                            str(restored_chemeleon_init),
                        ]
                        run(
                            restore_command,
                            logs_dir / "restore_chemeleon_encoder.log",
                            args.dry_run,
                        )

                    if "chemeleon_frozen" in requested_strategies:
                        chemeleon_frozen_dir = models_dir / "chemeleon_frozen"
                        checkpoints["chemeleon_frozen"] = ensure_trained(
                            train_command(
                                args,
                                paths["cumulative"],
                                chemeleon_frozen_dir,
                                model_targets,
                                data_seed,
                                pytorch_seed,
                                checkpoint=restored_chemeleon_init,
                                freeze_encoder=True,
                                featurizer_mode="V2",
                            ),
                            chemeleon_frozen_dir,
                            logs_dir / "train_chemeleon_frozen.log",
                            args.dry_run,
                        )

                    if "chemeleon_staged_unfreeze" in requested_strategies:
                        chemeleon_warmup_dir = models_dir / "chemeleon_staged_unfreeze_head_warmup"
                        chemeleon_warmup = ensure_trained(
                            train_command(
                                args,
                                paths["cumulative"],
                                chemeleon_warmup_dir,
                                model_targets,
                                data_seed,
                                pytorch_seed,
                                checkpoint=restored_chemeleon_init,
                                freeze_encoder=True,
                                featurizer_mode="V2",
                                epochs_override=args.staged_head_epochs,
                            ),
                            chemeleon_warmup_dir,
                            logs_dir / "train_chemeleon_staged_unfreeze_head_warmup.log",
                            args.dry_run,
                        )
                        chemeleon_staged_dir = models_dir / "chemeleon_staged_unfreeze"
                        checkpoints["chemeleon_staged_unfreeze"] = ensure_trained(
                            train_command(
                                args,
                                paths["cumulative"],
                                chemeleon_staged_dir,
                                model_targets,
                                data_seed,
                                pytorch_seed,
                                checkpoint=chemeleon_warmup,
                                featurizer_mode="V2",
                                epochs_override=args.epochs - args.staged_head_epochs,
                            ),
                            chemeleon_staged_dir,
                            logs_dir / "train_chemeleon_staged_unfreeze_full_model.log",
                            args.dry_run,
                        )

                continual_configs = {
                    "scratch_continual_cumulative": ("scratch", "cumulative", None),
                    "scratch_continual_replay": ("scratch", "replay", None),
                    "scratch_continual_similarity_replay": (
                        "scratch",
                        "similarity_replay",
                        None,
                    ),
                    "scratch_continual_similarity_matched_random": (
                        "scratch",
                        "similarity_matched_random",
                        None,
                    ),
                    "scratch_continual_incremental": ("scratch", "incremental", None),
                    "chemeleon_continual_cumulative": ("chemeleon_finetune", "cumulative", "V2"),
                    "chemeleon_continual_replay": ("chemeleon_finetune", "replay", "V2"),
                    "chemeleon_continual_similarity_replay": (
                        "chemeleon_finetune",
                        "similarity_replay",
                        "V2",
                    ),
                    "chemeleon_continual_similarity_matched_random": (
                        "chemeleon_finetune",
                        "similarity_matched_random",
                        "V2",
                    ),
                    "chemeleon_continual_incremental": (
                        "chemeleon_finetune",
                        "incremental",
                        "V2",
                    ),
                    "continual_cumulative": ("foundation_finetune", "cumulative", None),
                    "continual_replay": ("foundation_finetune", "replay", None),
                    "continual_similarity_replay": (
                        "foundation_finetune",
                        "similarity_replay",
                        None,
                    ),
                    "continual_similarity_matched_random": (
                        "foundation_finetune",
                        "similarity_matched_random",
                        None,
                    ),
                    "continual_incremental": ("foundation_finetune", "incremental", None),
                }
                requested_continual = [
                    strategy for strategy in continual_configs if strategy in requested_strategies
                ]
                if origin == 1:
                    for strategy in requested_continual:
                        base_strategy = continual_configs[strategy][0]
                        checkpoints[strategy] = checkpoints[base_strategy]
                else:
                    for strategy in requested_continual:
                        _, training_key, continual_featurizer = continual_configs[strategy]
                        if training_key == "replay":
                            strategy_train_path = replay_paths[(origin, data_seed)]
                        elif training_key == "similarity_replay":
                            strategy_train_path = similarity_prospective_paths[origin]
                        elif training_key == "similarity_matched_random":
                            strategy_train_path = similarity_matched_paths[(origin, data_seed)]
                        else:
                            strategy_train_path = paths[training_key]
                        model_dir = models_dir / strategy
                        checkpoints[strategy] = ensure_trained(
                            train_command(
                                args,
                                strategy_train_path,
                                model_dir,
                                model_targets,
                                data_seed,
                                pytorch_seed,
                                checkpoint=previous_continual[strategy],
                                featurizer_mode=continual_featurizer,
                            ),
                            model_dir,
                            logs_dir / f"train_{strategy}.log",
                            args.dry_run,
                        )

                for strategy in requested_strategies:
                    if "single_task" in strategy:
                        continue
                    prediction_path = predictions_dir / f"{strategy}_all_future.csv"
                    if origin == 1 and strategy in requested_continual:
                        base_strategy = continual_configs[strategy][0]
                        source = predictions_dir / f"{base_strategy}_all_future.csv"
                        if args.dry_run:
                            prediction_path.write_text(f"ALIAS {source}\n")
                        elif source.is_file() and not prediction_path.exists():
                            try:
                                prediction_path.symlink_to(source.name)
                            except OSError:
                                shutil.copy2(source, prediction_path)
                        elif not source.is_file():
                            # The canonical foundation prediction is created first below.
                            pass
                        continue
                    ensure_prediction(
                        args,
                        checkpoints[strategy],
                        paths["prediction_input"],
                        prediction_path,
                        logs_dir / f"predict_{strategy}.log",
                        featurizer_mode="V2" if strategy.startswith("chemeleon_") else None,
                    )
                # Create origin-1 aliases after the canonical prediction exists.
                if origin == 1 and not args.dry_run:
                    for strategy in requested_continual:
                        base_strategy = continual_configs[strategy][0]
                        source = predictions_dir / f"{base_strategy}_all_future.csv"
                        alias = predictions_dir / f"{strategy}_all_future.csv"
                        if not alias.exists():
                            try:
                                alias.symlink_to(source.name)
                            except OSError:
                                shutil.copy2(source, alias)

                previous_continual = {
                    strategy: checkpoints[strategy] for strategy in requested_continual
                }

    if args.dry_run:
        print(f"Dry run prepared under {output_dir}; no models or notebook were executed.")
        return
    if not args.no_analysis:
        analyzer = Path(__file__).with_name("analyze_temporal_experiment.py").resolve()
        command = [
            sys.executable,
            str(analyzer),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--practical-threshold-percent",
            str(args.practical_threshold_percent),
        ]
        run(command, output_dir / "analysis" / "analysis.log", False)
        notebook_builder = Path(__file__).with_name("build_results_notebook.py").resolve()
        run(
            [sys.executable, str(notebook_builder)],
            output_dir / "analysis" / "build_notebook.log",
            False,
        )
    print(f"Experiment complete: {output_dir}")


if __name__ == "__main__":
    main()
