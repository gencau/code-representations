"""
    This script combines two or more representations (file names, role summaries, tech summaries, raw source, bug report summaries)
    into a single ranking by selecting unique hits and preserving the order.
    It stops when the target top-k is reached.
"""
import argparse
import ast
import csv
import itertools
from pathlib import Path
from collections import Counter

def normalize_scores(files: list, scores: list, invert: bool = False) -> list:
    """Min-max normalize scores to [0,1]; invert if lower-is-better."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        normed = [1.0] * len(scores)
    else:
        normed = [(s - lo) / (hi - lo) for s in scores]
    if invert:
        normed = [1.0 - s for s in normed]
    return list(zip(files, normed))


def join_ranks(list_a: list, list_b: list, topk: int) -> list:
    """Priority-based merge: list_a items take precedence, then list_b fills gaps."""
    seen = set()
    results = []
    for hit in itertools.chain(list_a, list_b):
        if hit not in seen:
            seen.add(hit)
            results.append(hit)
        if len(results) == topk:
            break
    return results

def join_ranks_rrf(*lists: list, topk: int, k: int = 60) -> list:
    """Reciprocal Rank Fusion across any number of ranked lists."""
    scores = {}
    for ranked_list in lists:
        for rank, item in enumerate(ranked_list):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]

def join_ranks_roundrobin(*scored_lists: list, topk: int) -> list:
    """Round-robin merge with scores: alternates picks across lists by position,
    within each round sorts candidates by score ascending before adding."""
    seen = set()
    results = []
    for items in itertools.zip_longest(*scored_lists):
        # Collect (file, score) candidates from this round position, sorted by score
        candidates = sorted(
            (item for item in items if item is not None),
            key=lambda x: x[1],
            reverse=True
        )
        for file, _score in candidates:
            if file not in seen:
                seen.add(file)
                results.append(file)
        if len(results) >= topk:
            break
    return results[:topk]

def join_ranks_roundrobin_no_scores(*lists: list, topk: int) -> list:
    """Round-robin merge without scores: alternates picks across lists by position,
    within each round prioritizes items appearing in more lists (frequency weighting)."""
    seen = set()
    results = []
    freq = Counter(item for lst in lists for item in lst)
    for items in itertools.zip_longest(*lists):
        candidates = sorted(
            {item for item in items if item is not None and item not in seen},
            key=lambda x: freq[x],
            reverse=True
        )
        for file in candidates:
            seen.add(file)
            results.append(file)
        if len(results) >= topk:
            break
    return results[:topk]

def join_ranks_combsum(*scored_lists: list, topk: int) -> list:
    """CombSUM (Fox & Shaw, 1994 TREC-2): sum min-max-normalised scores across lists.
    Documents absent from a list contribute 0 for that list."""
    combined = {}
    for scored in scored_lists:
        for file, score in scored:
            combined[file] = combined.get(file, 0.0) + score
    return sorted(combined, key=combined.__getitem__, reverse=True)[:topk]

def join_ranks_simple(*lists: list, topk: int, k: int = 20) -> list:
    scores = {}
    score_counter = Counter()

    for ranked_list in lists:
        for loc in enumerate(ranked_list):
            score_counter[loc] += 1

    scores = [loc for loc, _ in score_counter.most_common()]

    return scores[:topk]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_list", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--method", type=str, choices=["priority", "rrf", "simple", "roundrobin", "rrb_llm", "combsum"], default="rrf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranked_results", type=bool, default=False)

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    csv.field_size_limit(10 ** 9)

    paths = args.results_list.split(',')

    def read_csv(path):
        with open(path) as f:
            return list(csv.DictReader(f))

    all_tables = [read_csv(p) for p in paths]

    base_rows = all_tables[0]
    fieldnames = list(base_rows[0].keys()) + ["fused_results"]

    output_rows = []
    for i, base_row in enumerate(base_rows):
        ranked_lists = []
        for table in all_tables:
            row = table[i]
            if args.ranked_results:
                key = "ranked_files" if "ranked_files" in row else "final_files"
            else:
                key = "topk_files" if "topk_files" in row else "final_files"
            ranked_lists.append(ast.literal_eval(row[key]))

        if args.method == "priority":
            if len(ranked_lists) != 2:
                raise ValueError("priority method requires exactly 2 lists")
            fused = join_ranks(ranked_lists[0], ranked_lists[1], args.topk)
        elif args.method == "rrf":
            fused = join_ranks_rrf(*ranked_lists, topk=args.topk)
        elif args.method in ("roundrobin", "combsum"):
            scored_lists = []
            for table, files in zip(all_tables, ranked_lists):
                row = table[i]
                if "rank_scores" in row:
                    scores = ast.literal_eval(row["rank_scores"])
                    invert = False  # higher is better
                elif "distances" in row:
                    scores = ast.literal_eval(row["distances"])
                    invert = True   # lower is better
                elif "distance" in row:
                    scores = ast.literal_eval(row["distance"])
                    invert = True   # lower is better
                else:
                    scores = [0.0] * len(files)
                    invert = False
                scored_lists.append(normalize_scores(files, scores, invert=invert))
            if args.method == "combsum":
                fused = join_ranks_combsum(*scored_lists, topk=args.topk)
            else:
                fused = join_ranks_roundrobin(*scored_lists, topk=args.topk)
        elif args.method == "rrb_llm":
            fused = join_ranks_roundrobin_no_scores(*ranked_lists, topk=args.topk)
        else:
            fused = join_ranks_simple(*ranked_lists, topk=args.topk, k=20)

        output_row = dict(base_row)
        output_row["fused_results"] = fused
        output_rows.append(output_row)

    output_path = args.output / "results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

if __name__ == "__main__":
    main()