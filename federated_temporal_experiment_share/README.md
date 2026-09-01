# Federated temporal modelling — portable experiment

This archive is a self-contained hand-off of the pKi, LogD, and solubility temporal-modelling study. It deliberately excludes the thousands of raw prediction files and all trained checkpoints from the completed study.

## What to open

- **`RESULTS_WORKBOOK.ipynb`** — the completed, executed workbook. It can be read immediately. If you rerun it, its first cell automatically expands `results_snapshot.zip`, which contains only the compact analysis tables needed by the workbook.
- **`RUN_EXPERIMENT.ipynb`** — the simple control workbook for reproducing the experiment. Its configuration cell supports the complete study or a selected strategy subset.

The two source inputs are under `data/`. Implementation details are under `pipeline/`; normally there is no need to edit them.

## One-time setup

From a terminal in this extracted directory:

```bash
conda env create -f environment.yml
conda activate chemprop
jupyter lab RUN_EXPERIMENT.ipynb
```

Run all cells in `RUN_EXPERIMENT.ipynb`. The complete setting reproduces all registered methods using four rolling origins and 25 crossed data/PyTorch seed combinations per origin. This is a substantial multi-day computation on a laptop. Chemprop downloads the public CheMeleon weights on first use, so internet access is required for that first CheMeleon run.

The equivalent terminal command is:

```bash
conda activate chemprop
python pipeline/portable_run.py
```

## Experimental safeguards

- Five contiguous temporal buckets are optimized for balanced row counts, but a date is never split between buckets.
- Four rolling-origin challenges train only on data available up to that origin.
- Chemprop receives `--split-sizes 0.8 0.2 0.0`, `--epochs 100`, and `--patience 10`.
- pKi and LogD stay on their existing logarithmic scales. Solubility is modelled and evaluated as `log10(1 + uM)`.
- Five data seeds are crossed with five PyTorch seeds, producing 25 matched fits at each origin.
- Test-row bootstrap resampling supplies 95% uncertainty intervals; bootstrap draws are not treated as independent experimental units.
- Primary claims use the next temporal fold. All-future and final-fold views are reported separately as sensitivity analyses.
- The supplied federated model, scratch models, and CheMeleon receive symmetric applicable strategy variants. The final north-star comparison selects the best federation and best alternative without using the held-out origin being evaluated.

## Disk use and recovery

The runner is resumable while checkpoints are retained. Redundant Lightning checkpoints are pruned immediately after every successful Chemprop fit. Once predictions, statistical analysis, and the final workbook have all been verified, trained models are removed by default; predictions and analysis remain. Use `--keep-models` only if you intend to add methods incrementally and have ample disk space.

An interrupted model directory is moved into `temporal_experiment/incomplete_archive/` and retried automatically. Three repeated interruptions of the same fit are treated as a real error and stop the run.

## Principal outputs after a run

- `RESULTS_WORKBOOK.ipynb` — newly executed results workbook
- `run_status.json` — completion and verification status
- `experiment_run.log` and `workbook_execution.log` — reproducibility logs
- `temporal_experiment/analysis/` — metrics, bootstrap intervals, significance tests, diagnostics, and plotting data
- `temporal_experiment/origin_*/.../predictions/` — raw Chemprop predictions

The source dataset and supplied checkpoint are intentionally included. Share this archive only with colleagues authorized to receive those materials.
