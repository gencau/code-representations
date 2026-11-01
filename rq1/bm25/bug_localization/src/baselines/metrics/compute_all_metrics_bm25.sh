#!/usr/bin/env bash
# Recompute RQ1 BM25 metrics at k=5 for every representation used in the paper.
# Reads results from all_results/ and writes per-task CSVs to the same relative
# path under all_results_full/, plus one row per split to all_results_full/metrics_summary.csv.
#
# Usage: bash src/baselines/metrics/compute_all_metrics_bm25.sh   (PYTHON=... to pick an interpreter)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"   # code-representations/
PYTHON="${PYTHON:-python}"
TOPK=5
RESULTS=rq1-bm25_embeddings

# dataset representation relative-path-under-all_results
RUNS=(
  "lca project_structure  $RESULTS/filepaths/lca/bm25-retriever_0/results.csv"
  "lca role_summaries     $RESULTS/prompt-variations/lca/all/bm25-retriever_0_100_percent_prompt2_all/results.csv"
  "lca detailed_summaries $RESULTS/prompt-variations/lca/all/bm25-retriever_0_100_percent_prompt3_all/results.csv"
  "lca bug_reports        $RESULTS/queries/lca/bm25-retriever_0_100_percent_5_queries_prompt3_all/results.csv"
  "lca raw_sources        $RESULTS/raw-code/lca/results_bm25-lca.csv"
  "swe project_structure  $RESULTS/filepaths/swe/bm25-retriever_0/results.csv"
  "swe role_summaries     $RESULTS/prompt-variations/swe/all/bm25-retriever_0_100_percent_prompt2_all/results.csv"
  "swe detailed_summaries $RESULTS/prompt-variations/swe/all/bm25-retriever_0_100_percent_prompt3_all/results.csv"
  "swe bug_reports        $RESULTS/queries/swe/bm25-retriever_0_100_percent_queries_all/results.csv"
  "swe raw_sources        $RESULTS/raw-code/swe/results_bm25-swe.csv"
)

for run in "${RUNS[@]}"; do
  read -r dataset representation path <<< "$run"
  echo "==== $dataset / $representation"
  "$PYTHON" "$SCRIPT_DIR/compute_metrics_bm25.py" \
    --input "$ROOT/all_results/$path" \
    --output "$ROOT/all_results_full/$path" \
    --summary "$ROOT/all_results_full/metrics_summary.csv" \
    --topk "$TOPK" \
    --rq rq1 --model bm25 --representation "$representation" --dataset "$dataset"
done
