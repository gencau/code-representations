"""
Shared output handling for the compute_metrics scripts: per-task labels,
per-split summaries (all / single-file / multi-file) and the combined
all_results_full/metrics_summary.csv.
"""
import os

import pandas as pd

LABELS = ['rq', 'model', 'representation', 'dataset']
SUMMARY_KEY = LABELS + ['split', 'k']

# summary column -> per-task column written by the compute_metrics scripts
SUMMARY_METRICS = {
    'avg_preds': 'n_preds',
    'MAP@k': 'MAP',
    'MRR@k': 'MRR',
    'Hit@1': 'hit_rate@1',
    'Hit@3': 'hit_rate@3',
    'Hit@k': 'hit_rate@k',
    'Recall@k': 'recall@k',
    'Acc@k': 'Acc@k',
    'Precision@k': 'precision@k',
}


def add_output_arguments(parser):
    parser.add_argument("--output", required=True,
                        help="Per-task CSV to write (never the same file as --input)")
    parser.add_argument("--summary", default=None,
                        help="Shared summary CSV to add one row per split to")
    for name in LABELS:
        parser.add_argument(f"--{name}", default=None)


def labels_from_args(args) -> dict:
    return {name: getattr(args, name) for name in LABELS}


def check_output(input_file: str, output_file: str):
    if os.path.abspath(output_file) == os.path.abspath(input_file):
        raise ValueError("--output must differ from --input; results under all_results/ are read-only")


def task_type(gt_files) -> str:
    return 'single' if len(set(gt_files)) == 1 else 'multi'


def summarize_splits(df: pd.DataFrame, gt_column: str,
                     column_prefix: str = "", column_suffix: str = "") -> pd.DataFrame:
    """One row per split (all / single / multi) with the mean of each per-task metric."""
    types = df[gt_column].apply(task_type)
    splits = {'all': df, 'single': df[types == 'single'], 'multi': df[types == 'multi']}

    summary = []
    for split, subset in splits.items():
        row = {'split': split, 'n': len(subset)}
        for name, col in SUMMARY_METRICS.items():
            row[name] = subset[column_prefix + col + column_suffix].dropna().mean()
        summary.append(row)
    return pd.DataFrame(summary)


def save_results(df: pd.DataFrame, output: str, gt_column: str, labels: dict):
    """Write the per-task CSV with label and task_type columns."""
    for name in LABELS:
        df[name] = labels.get(name)
    df['task_type'] = df[gt_column].apply(task_type)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df.to_csv(output, index=False)


def upsert_summary(summary_file: str, summary: pd.DataFrame, labels: dict, k_label: str):
    """Add rows to the shared summary CSV, replacing earlier rows with the same key."""
    summary = summary.copy()
    summary.insert(0, 'k', k_label)
    for i, name in enumerate(LABELS):
        summary.insert(i, name, labels.get(name))

    if os.path.exists(summary_file):
        existing = pd.read_csv(summary_file, dtype={'k': str})
        summary = pd.concat([existing, summary], ignore_index=True)
    summary = summary.drop_duplicates(subset=SUMMARY_KEY, keep='last')
    summary = summary.sort_values(SUMMARY_KEY, kind='stable')
    # Keep a fixed column order when metrics are added to SUMMARY_METRICS later
    ordered = LABELS + ['k', 'split', 'n'] + list(SUMMARY_METRICS)
    summary = summary[ordered + [c for c in summary.columns if c not in ordered]]
    os.makedirs(os.path.dirname(os.path.abspath(summary_file)), exist_ok=True)
    summary.to_csv(summary_file, index=False)
