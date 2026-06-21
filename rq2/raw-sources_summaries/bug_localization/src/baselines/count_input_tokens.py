"""
Count input tokens from HFDataSource or SWEDataSource.

The "input" for each data point is:
  - issue description  (issue_title + issue_body for HF; problem_statement for SWE)
  - repo content       (sum of all non-test source file contents)

Token counts use the openai/gpt-oss-20b tokenizer (o200k_harmony encoding).

Usage:
    # HF dataset
    python count_input_tokens.py --source hf \
        --hub_name tiginamaria/bug-localization \
        --repos_dir /path/to/repos \
        --configs py java kt \
        --split test

    # SWE-bench dataset (local parquet/json path or HF hub name)
    python count_input_tokens.py --source swe \
        --hub_name /path/to/swe_bench.parquet \
        --repos_dir /path/to/repos

    # Write per-item counts to a CSV
    python count_input_tokens.py --source hf ... --output counts.csv
"""

import argparse
import csv
import os
import sys
from dotenv import load_dotenv

from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.baselines.data_sources.hf_data_source import HFDataSource
from src.baselines.data_sources.swe_bench_data_source import SWEDataSource


def _issue_text_hf(dp: dict) -> str:
    return dp.get("issue_title", "") + "\n" + dp.get("issue_body", "")


def _issue_text_swe(dp: dict) -> str:
    return dp.get("problem_statement", "")


def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def _count_repo_tokens(repo_content: dict, enc) -> int:
    return sum(
        _count_tokens(content, enc)
        for content in repo_content.values()
        if content
    )

def _count_repo_path_tokens(repo_content: dict, enc) -> int:
    return sum(
        _count_tokens(path, enc)
        for path in repo_content.keys()
        if path
    )


def count_tokens_for_source(data_source, get_issue_text, enc, representation: str, output_csv: str | None):
    rows = []
    total_issue = 0
    total_repo = 0

    for i, (dp, repo_content, _changed_files) in enumerate(data_source):
        issue_tokens = _count_tokens(get_issue_text(dp), enc)

        if representation == "raw":
            repo_tokens = _count_repo_tokens(repo_content, enc)
        else:
            repo_tokens = _count_repo_path_tokens(repo_content, enc)
        total_issue += issue_tokens
        total_repo += repo_tokens

        instance_id = dp.get("instance_id") or dp.get("repo_owner", "") + "/" + dp.get("repo_name", "")
        rows.append({
            "instance_id": instance_id,
            "issue_tokens": issue_tokens,
            "repo_tokens": repo_tokens,
            "total_tokens": issue_tokens + repo_tokens,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} items...", flush=True)

    if output_csv:
        try:
            os.makedirs(output_csv, exist_ok=True)
            full_path = os.path.join(output_csv, "results.csv")
            with open(full_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["instance_id", "issue_tokens", "repo_tokens", "total_tokens"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Per-item counts saved to '{output_csv}'")
        except Exception:
            pass

    return rows, total_issue, total_repo


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Count input tokens from HFDataSource or SWEDataSource.")
    parser.add_argument("--source", choices=["hf", "swe"], required=True, help="Data source type")
    parser.add_argument("--hub_name", required=True, help="HF hub name or local dataset path")
    parser.add_argument("--repos_dir", required=True, help="Directory containing cloned repositories")
    parser.add_argument("--configs", nargs="*", default=None, help="Dataset config names (HF only)")
    parser.add_argument("--split", default=None, help="Dataset split (e.g. 'test')")
    parser.add_argument("--cache_dir", default=None, help="HF datasets cache directory")
    parser.add_argument("--output", default=None, help="Optional path to write per-item CSV")
    parser.add_argument("--representation", default="raw", help="Type of representation: raw, filepaths")
    args = parser.parse_args()

    enc = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    if args.source == "hf":
        data_source = HFDataSource(
            hub_name=args.hub_name,
            repos_dir=args.repos_dir,
            configs=args.configs,
            split=args.split,
            cache_dir=args.cache_dir,
        )
        get_issue_text = _issue_text_hf
        label = "HFDataSource"
    else:
        data_source = SWEDataSource(
            hub_name=args.hub_name,
            repos_dir=args.repos_dir,
            configs=args.configs,
            split=args.split,
            cache_dir=args.cache_dir,
        )
        get_issue_text = _issue_text_swe
        label = "SWEDataSource"

    print(f"Counting input tokens for {label} ({args.hub_name})...")
    rows, total_issue, total_repo = count_tokens_for_source(data_source, get_issue_text, enc, args.representation, args.output)

    n = len(rows)
    total = total_issue + total_repo

    print(f"\n=== Results ({label}) ===")
    print(f"  Data points        : {n:,}")
    print(f"  Issue tokens       : {total_issue:,}")
    print(f"  Repo tokens        : {total_repo:,}")
    print(f"  Total input tokens : {total:,}")
    if n > 0:
        print(f"  Avg tokens / item  : {total / n:,.0f}")


if __name__ == "__main__":
    main()
