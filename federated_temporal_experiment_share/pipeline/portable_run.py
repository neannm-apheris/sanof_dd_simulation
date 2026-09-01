#!/usr/bin/env python3
"""Run, verify, and disk-clean the portable temporal experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_ROOT / "temporal_experiment"
STATUS_PATH = PACKAGE_ROOT / "run_status.json"
SOURCE_NOTEBOOK = PACKAGE_ROOT / "temporal_results_explorer.ipynb"
FINAL_NOTEBOOK = PACKAGE_ROOT / "RESULTS_WORKBOOK.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Optional strategy subset. Omit to run the complete registered experiment.",
    )
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Keep trained models for later incremental additions (uses much more disk space).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace a prior local experiment run.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**values: object) -> None:
    STATUS_PATH.write_text(json.dumps(values, indent=2) + "\n")


def run(command: list[str], log_path: Path) -> None:
    print("$", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / ".runtime_cache" / "matplotlib"))
    environment.setdefault("XDG_CACHE_HOME", str(PACKAGE_ROOT / ".runtime_cache" / "xdg"))
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=PACKAGE_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def discard_extracted_snapshot_if_present() -> None:
    marker = RESULTS_DIR / "SNAPSHOT_ONLY.json"
    if marker.is_file():
        if list(RESULTS_DIR.glob("origin_*")):
            raise RuntimeError(
                "The analysis snapshot marker and raw experiment origins are both present; "
                "refusing to guess which data to replace."
            )
        shutil.rmtree(RESULTS_DIR)


def verify_results() -> tuple[dict[str, int], int]:
    manifest = json.loads((RESULTS_DIR / "experiment_manifest.json").read_text())
    expected = (
        len(manifest["rolling_origins"])
        * len(manifest["data_seeds"])
        * len(manifest["pytorch_seeds"])
    )
    counts = {
        strategy: len(
            list(
                RESULTS_DIR.glob(
                    f"origin_*/replicates/*/predictions/{strategy}_all_future.csv"
                )
            )
        )
        for strategy in manifest["strategies"]
    }
    incomplete = {strategy: count for strategy, count in counts.items() if count != expected}
    if incomplete:
        raise RuntimeError(f"Prediction coverage is incomplete: {incomplete}")

    notebook = nbformat.read(FINAL_NOTEBOOK, as_version=4)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"Final notebook contains {len(errors)} execution error(s)")
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    executed_cells = sum(
        cell.cell_type == "code" and cell.get("execution_count") is not None
        for cell in notebook.cells
    )
    if executed_cells != code_cells:
        raise RuntimeError(f"Only {executed_cells}/{code_cells} notebook cells executed")
    return counts, code_cells


def remove_models() -> tuple[int, int]:
    model_roots = list(RESULTS_DIR.glob("origin_*/replicates/*/models"))
    reclaimed = sum(
        path.stat().st_size
        for root in model_roots
        for path in root.rglob("*")
        if path.is_file()
    )
    for root in model_roots:
        shutil.rmtree(root)
    return len(model_roots), reclaimed


def main() -> None:
    args = parse_args()
    try:
        discard_extracted_snapshot_if_present()
        write_status(state="running_experiment", started_at=utc_now())
        experiment_command = [
            sys.executable,
            str(PIPELINE_DIR / "run_temporal_experiment.py"),
        ]
        if args.strategies:
            experiment_command.extend(["--strategies", *args.strategies])
        if args.overwrite:
            experiment_command.append("--overwrite")
        run(experiment_command, PACKAGE_ROOT / "experiment_run.log")

        jupyter = shutil.which("jupyter")
        if jupyter is None:
            raise RuntimeError("The active environment does not provide the jupyter command")
        write_status(state="executing_results_workbook", updated_at=utc_now())
        run(
            [
                jupyter,
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(SOURCE_NOTEBOOK),
                "--output",
                FINAL_NOTEBOOK.name,
                "--output-dir",
                str(PACKAGE_ROOT),
                "--ExecutePreprocessor.timeout=1800",
            ],
            PACKAGE_ROOT / "workbook_execution.log",
        )
        counts, code_cells = verify_results()

        removed = 0
        reclaimed = 0
        if not args.keep_models:
            removed, reclaimed = remove_models()
        write_status(
            state="complete",
            completed_at=utc_now(),
            strategies=len(counts),
            predictions=sum(counts.values()),
            predictions_per_strategy=counts,
            notebook=str(FINAL_NOTEBOOK),
            notebook_code_cells=code_cells,
            notebook_error_outputs=0,
            model_directories_removed=removed,
            model_bytes_reclaimed=reclaimed,
            models_retained=args.keep_models,
        )
        print(f"Complete. Open {FINAL_NOTEBOOK}", flush=True)
    except Exception as error:
        write_status(
            state="failed",
            failed_at=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
