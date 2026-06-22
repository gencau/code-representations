'''
This script analyzes the found files across multiple CSV files, comparing them against the ground truth files. It generates two output CSV files:
1. per_df_buglists.csv: Contains one row per dataframe with lists of bugs solved, uniquely solved, and fully solved by that dataframe.
2. bugs_solved_matrix.csv: Contains one row per bug with a binary matrix indicating which dataframes solved it.'''

import re

import pandas as pd
import os, ast
from collections import defaultdict
import unidiff
from typing import List

import argparse

index_column = "index"
found_files_column = "topk_files"
ground_truth_column = "expected_files"

TEST_DIR_NAMES = {"test", "tests", "testing"}

def _unidiff_files(diff_str: str) -> List[str]:
    """
    Return file paths from a git diff in the exact order they appear,
    excluding anything located in a test/tests/testing directory.
    Duplicates are removed while preserving first occurrence.
    """
    files_in_order: List[str] = []
    seen = set()

    patch = _regex_files(diff_str)

    for patched_file in patch:
        # Strip leading "a/" if present (Git diff format)
        path = patched_file.split("a/", 1)[-1]

        # Skip if any directory component is test/tests/testing (case-insensitive)
        if any(part.lower() in TEST_DIR_NAMES for part in path.split('/')):
            continue

        if path not in seen:
            seen.add(path)
            files_in_order.append(path)

    return files_in_order

_PATH_RE = re.compile(r'^\+\+\+\s+b/(.+)$', re.MULTILINE)

def _regex_files(diff: str) -> list[str]:
    return list(dict.fromkeys(_PATH_RE.findall(diff)))   # dedupe + preserve order

def __main__():
    parser = argparse.ArgumentParser(description="Analyze found files")
    parser.add_argument("--data-paths", type=str, help="Path to the CSV files, comma-separated")
    parser.add_argument("--topk", type=int, default=5, help="Top K files to consider")
    parser.add_argument("--type", type=str, default="embedding", help="BM25, embedding, llm, swe, fused")
    args = parser.parse_args()

    if not args.data_paths:
        print("Please provide a path to the CSV file.")
        return
    
    if args.type.lower() == "bm25" or args.type.lower() == "llm":
        global index_column, found_files_column, ground_truth_column
        index_column = "id"
        found_files_column = "final_files"
        ground_truth_column = "changed_files"
    elif args.type.lower() == "fused":
        index_column = "index"
        found_files_column = "fused_results"
        ground_truth_column = "expected_files"
    elif args.type.lower() == "swe":
        index_column = "instance_id"
        found_files_column = "final_files"
        # If ground truth column not provided, extract it from the patch column, which is a git patch format
        ground_truth_column = "changed_files"
        # changed files doesn't exist for swe, create it by parsing the patch column
        for path in args.data_paths.split(","):
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                continue
            df = pd.read_csv(path)
            if found_files_column not in df.columns:
                found_files_column = "topk_files"
            if index_column not in df.columns:
                index_column = "index"
    
                
            if "changed_files" not in df.columns:
                if "expected_files" not in df.columns:
                    print(f"Extracting changed files for {path}...")
                    df["changed_files"] = df["patch"].apply(_unidiff_files)
                    df.to_csv(path, index=False)
                    print(f"Updated {path} with changed_files column.")
                else:
                    ground_truth_column = "expected_files"
        
        
    df_list = []
    for path in args.data_paths.split(","):
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            continue
        df = pd.read_csv(path)
        df_list.append(df)

    # Now get the ground truth files from the first dataframe
    if not df_list:
        print("No valid CSV files provided.")
        return
    
    # Now check, in each dataframe, which files were found in the final_files column and for each row, compare against ground truth
    all_df_found_files = list()
    all_df_ground_truth_files = list()
    for i, df in enumerate(df_list):
        found_files = set()
        ground_truth_files = set()
        for final_files_str in df[found_files_column].dropna().tolist():
            final_files = eval(final_files_str)  # Convert string representation of list to actual list
            found_files.update(final_files[:args.topk])
            ground_truth_files.update(eval(df.loc[df[found_files_column] == final_files_str, ground_truth_column].values[0]))
        all_df_found_files.append(found_files)
        all_df_ground_truth_files.append(ground_truth_files)
        print(f"DataFrame {i}: Found {len(found_files)} unique files.")

    # For each dataframe, print the set of files found which are also in ground truth
    for found_files, ground_truth_files in zip(all_df_found_files, all_df_ground_truth_files):
        print(f"\nDataFrame {i} found files:")
        for f in found_files:
            if f in ground_truth_files:
                print(f)

    # Now do a verification on the 'id' column, to make sure all dataframes have the same ids
    base_ids = set(df_list[0][index_column].tolist())
    for i, df in enumerate(df_list[1:], start=1):
        current_ids = set(df[index_column].tolist())
        if base_ids != current_ids:
            print(f"DataFrame 0 and DataFrame {i} have different index_column values.")
            missing_in_current = base_ids - current_ids
            missing_in_base = current_ids - base_ids
            if missing_in_current:
                print(f"  IDs in DataFrame 0 but not in DataFrame {i}: {missing_in_current}")
            if missing_in_base:
                print(f"  IDs in DataFrame {i} but not in DataFrame 0: {missing_in_base}")
        else:
            print(f"DataFrame 0 and DataFrame {i} have the same index_column values.")


    # Map bug_id -> set of dataframe indices that solved it
    solvers = defaultdict(set)           # bug_id -> set(df indices that solved it)
    hit_counts = [0] * len(df_list)
    any_file_counts = [0] * len(df_list)

    all_bug_ids = set()

    for i, df in enumerate(df_list):
        for _, row in df.iterrows():
            bug_id = row[index_column]
            all_bug_ids.add(bug_id)

            expected_files = set(ast.literal_eval(row[ground_truth_column]))
            topk_files = list(ast.literal_eval(row[found_files_column]))[:args.topk]

            if len(topk_files) > 0:
                any_file_counts[i] += 1

            if expected_files & set(topk_files):
                hit_counts[i] += 1
                solvers[bug_id].add(i)

    # Build solved-by dataframe lists (ALL solved bugs, not just unique)
    solved_by_df = {i: [] for i in range(len(df_list))}
    for bug_id, idxs in solvers.items():
        for i in idxs:
            solved_by_df[i].append(bug_id)

    # Build the list of bugs for which all files in expected were found by a single dataframe
    all_files_found_by_df = {i: [] for i in range(len(df_list))}
    for bug_id, idxs in solvers.items():
        for i in idxs:
            df = df_list[i]
            row = df[df[index_column] == bug_id].iloc[0]
            expected_files = set(ast.literal_eval(row[ground_truth_column]))
            topk_files = list(ast.literal_eval(row[found_files_column]))[:args.topk]
            if expected_files.issubset(set(topk_files)):
                all_files_found_by_df[i].append(bug_id)

    # Unique lists (already implied by solvers, but build cleanly)
    unique_by_df = {i: [] for i in range(len(df_list))}
    for bug_id, idxs in solvers.items():
        if len(idxs) == 1:
            only_i = next(iter(idxs))
            unique_by_df[only_i].append(bug_id)

    # Sort lists for stable output
    for i in range(len(df_list)):
        solved_by_df[i] = sorted(solved_by_df[i])
        unique_by_df[i] = sorted(unique_by_df[i])

    # --- CSV 1: one row per dataframe, with bug-id lists ---
    rows = []
    for i in range(len(df_list)):
        rows.append({
            "dataframe": i,
            "bugs_total_rows": int(len(df_list[i])),
            "at_least_one_file_returned": int(any_file_counts[i]),
            "total_bugs_with_at_least_one_hit": int(len(solved_by_df[i])),
            "unique_bugs_fixed": int(len(unique_by_df[i])),
            "bugs_fixed_list": ";".join(map(str, solved_by_df[i])),
            "unique_bugs_fixed_list": ";".join(map(str, unique_by_df[i])),
            "all_files_found_by_one_df": ";".join(map(str, all_files_found_by_df[i])),
        })

    per_df_buglists = pd.DataFrame(rows)
    per_df_buglists.to_csv("results/per_df_buglists.csv", index=False)
    print("Wrote: per_df_buglists.csv")
    print(per_df_buglists)

    bug_rows = []
    for bug_id in sorted(all_bug_ids):
        idxs = solvers.get(bug_id, set())
        row = {"bug_id": bug_id, "n_solvers": len(idxs)}
        for i in range(len(df_list)):
            row[f"solved_by_df_{i}"] = 1 if i in idxs else 0
        bug_rows.append(row)

    bugs_solved_matrix = pd.DataFrame(bug_rows)
    bugs_solved_matrix.to_csv("results/bugs_solved_matrix.csv", index=False)
print("Wrote: bugs_solved_matrix.csv")


if __name__ == "__main__":
    __main__()