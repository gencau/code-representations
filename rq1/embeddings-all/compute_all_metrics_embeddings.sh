#!/usr/bin/env bash
# Recompute RQ1 embedding metrics at k=5 for every model and representation used in the paper.
# Reads results from all_results/ and writes per-task CSVs to the same relative
# path under all_results_full/, plus one row per split to all_results_full/metrics_summary.csv.
#
# Usage: bash compute_all_metrics_embeddings.sh   (PYTHON=... to pick an interpreter)
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # code-representations/
PYTHON="${PYTHON:-python}"
MAX_K=5
RESULTS=rq1-bm25_embeddings

# model label:filename token
MODELS="gte-large:gtelarge bge-large:bgelarge codexembed:codexembed qwen3:qwen"

# dataset representation glob-under-all_results/$RESULTS ({m} = model token)
RUNS=(
  "lca project_structure  filepaths/lca/results_*_{m}*.csv"
  "lca role_summaries     prompt-variations/lca/all/results_*prompt2[-_]{m}*.csv"
  "lca detailed_summaries prompt-variations/lca/all/results_*prompt3[-_]{m}*.csv"
  "lca bug_reports        queries/lca/results_*-prompt3-{m}*.csv"
  "lca raw_sources        raw-code/lca/results_2*_{m}*100percent.csv"
  "swe project_structure  filepaths/swe/results_*_{m}*.csv"
  "swe role_summaries     prompt-variations/swe/all/results_*prompt2[-_]{m}*.csv"
  "swe detailed_summaries prompt-variations/swe/all/results_*prompt3[-_]{m}*.csv"
  "swe bug_reports        queries/swe/results_*_queries_{m}*.csv"
  "swe raw_sources        raw-code/swe/results_2*_{m}*100percent.csv"
)

for run in "${RUNS[@]}"; do
  read -r dataset representation pattern <<< "$run"
  for entry in $MODELS; do
    model="${entry%%:*}"
    token="${entry##*:}"
    matches=( "$ROOT/all_results/$RESULTS/"${pattern//\{m\}/$token} )
    if [ "${#matches[@]}" -ne 1 ]; then
      echo "Expected one file for $dataset/$representation/$model, found ${#matches[@]}: ${matches[*]:-}" >&2
      exit 1
    fi
    input="${matches[0]}"
    echo "==== $dataset / $representation / $model: ${input#$ROOT/all_results/}"
    "$PYTHON" "$SCRIPT_DIR/compute_metrics.py" \
      --input "$input" \
      --output "$ROOT/all_results_full/${input#$ROOT/all_results/}" \
      --summary "$ROOT/all_results_full/metrics_summary.csv" \
      --max_k "$MAX_K" \
      --rq rq1 --model "$model" --representation "$representation" --dataset "$dataset"
  done
done
