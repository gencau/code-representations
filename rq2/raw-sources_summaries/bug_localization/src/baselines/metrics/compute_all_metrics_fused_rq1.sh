#!/usr/bin/env bash
# Rebuild the RQ1 fused rankings (two representations, RRF, top-5) from the same
# inputs as the RQ1 retrieval table, then compute metrics at k=5.
# Writes per-task CSVs to all_results_full/rq1-bm25_embeddings/fused_rankings/<dataset>/<model>/<folder>/
# and one row per split to all_results_full/metrics_summary.csv.
#
# Usage: bash compute_all_metrics_fused_rq1.sh   (PYTHON=... to pick an interpreter)
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"   # code-representations/
PYTHON="${PYTHON:-python}"
TOPK=5
RESULTS="$ROOT/all_results/rq1-bm25_embeddings"
FUSED_OUT="$ROOT/all_results_full/rq1-bm25_embeddings/fused_rankings"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# model label:output folder:filename token (empty token = BM25)
MODELS="bm25:bm25: gte-large:gtelarge:gtelarge bge-large:bgelarge:bgelarge codexembed:codexembed:codexembed qwen3:qwen3:qwen"

# first input:second input:output folder:representation label (the first input's rows are the base)
PAIRS="raw:path:rrf_raw_path:raw_sources+project_structure
raw:sum2:rrf_raw_sum2:raw_sources+role_summaries
raw:sum3:rrf_raw_sum3:raw_sources+detailed_summaries
path:sum2:rrf_path_sum2:project_structure+role_summaries
path:sum3:rrf_path_sum3:project_structure+detailed_summaries
sum2:sum3:rrf_sum23:role_summaries+detailed_summaries"

find_one() {
  local matches=( $1 )
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected one file for $1, found ${#matches[@]}: ${matches[*]:-}" >&2
    exit 1
  fi
  echo "${matches[0]}"
}

# input file for one dataset / representation / model token
input_for() {
  local ds=$1 rep=$2 t=$3
  if [ -z "$t" ]; then
    case $rep in
      raw)  find_one "$RESULTS/raw-code/$ds/results_bm25-$ds.csv" ;;
      path) find_one "$RESULTS/filepaths/$ds/bm25-retriever_0/results.csv" ;;
      sum2) find_one "$RESULTS/prompt-variations/$ds/all/bm25-retriever_0_100_percent_prompt2_all/results.csv" ;;
      sum3) find_one "$RESULTS/prompt-variations/$ds/all/bm25-retriever_0_100_percent_prompt3_all/results.csv" ;;
    esac
  else
    case $rep in
      raw)  find_one "$RESULTS/raw-code/$ds/results_2*_$t*100percent.csv" ;;
      path) find_one "$RESULTS/filepaths/$ds/results_*_$t*.csv" ;;
      sum2) find_one "$RESULTS/prompt-variations/$ds/all/results_*prompt2[-_]$t*.csv" ;;
      sum3) find_one "$RESULTS/prompt-variations/$ds/all/results_*prompt3[-_]$t*.csv" ;;
    esac
  fi
}

for dataset in lca swe; do
  for entry in $MODELS; do
    IFS=: read -r model folder token <<< "$entry"
    for pair in $PAIRS; do
      IFS=: read -r first second fused_name representation <<< "$pair"
      first_file="$(input_for "$dataset" "$first" "$token")"
      second_file="$(input_for "$dataset" "$second" "$token")"
      echo "==== $dataset / $model / $representation"
      echo "     ${first_file#$RESULTS/} + ${second_file#$RESULTS/}"

      fused_dir="$TMP/$dataset/$folder/$fused_name"
      "$PYTHON" "$SCRIPT_DIR/../combine_representations.py" \
        --results_list "$first_file,$second_file" --method rrf --topk "$TOPK" --output "$fused_dir"
      "$PYTHON" "$SCRIPT_DIR/compute_metrics_combined.py" \
        --input "$fused_dir/results.csv" \
        --output "$FUSED_OUT/$dataset/$folder/$fused_name/results.csv" \
        --summary "$ROOT/all_results_full/metrics_summary.csv" \
        --topk "$TOPK" \
        --rq rq1 --model "$model" --representation "$representation" --dataset "$dataset"
    done
  done
done
