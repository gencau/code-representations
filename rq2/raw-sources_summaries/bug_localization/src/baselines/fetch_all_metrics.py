#!/usr/bin/env python3
"""
collect_metrics.py  -run every results.csv twice (top-k = 5,10)
and capture Overall Average Precision + Overall Hit@k only.
"""

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

# ------------ regexes for the two wanted metrics ----------------------------
PATTERNS = {
    # Overall Average Precision: 0.1054…
    "avg_precision": re.compile(
        r"overall\s+average\s+precision\s*:\s*([0-9]*\.?[0-9]+)", re.I
    ),
    # Overall Hit@k: 0.226…
    "hit": re.compile(
        r"overall\s+hit@\s*(?:\d+|k)\s*:\s*([0-9]*\.?[0-9]+)", re.I
    ),
}


def parse_metrics(stdout: str) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for key, pat in PATTERNS.items():
        m = pat.search(stdout)
        if not m:
            raise ValueError(f"Could not find {key} in output:\n{stdout}")
        vals[key] = float(m.group(1))
    return vals


d# --- replace the existing run_eval function with this one ---------------

def run_eval(eval_script: Path, csv_path: Path, topk: int) -> Dict[str, float]:
    """
    Call `compute_metrics_llm.py` on one results.csv with:

        +output_path   parent directory of the run
        +name          run‑directory name
        +topk          {5,10}
        ++backbone.experiment="rerank"   (unless path contains 'project-structure')
    """
    run_dir = csv_path.parent           # …/<run_dir>/results.csv
    name = run_dir.name
    output_path = str(run_dir.parent)

    cmd = [
        "python",
        str(eval_script),
        f"+output_path={output_path}",
        f"+name={name}",
        f"+topk={topk}",
    ]

    # Add rerank override unless the path includes "project-structure"
    if "project-structure" not in str(csv_path):
        cmd.append('++backbone.experiment="rerank"')   # double‑plus for new key

    print("   ↳", " ".join(cmd))   # optional echo for debugging

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return parse_metrics(proc.stdout)



def write_rows(out_csv: Path, rows: Sequence[Dict[str, str]]) -> None:
    header = [
        "filename",
        "avg_precision5", "hit5",
        "avg_precision10", "hit10",
    ]
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------- main -------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, type=Path)
    ap.add_argument("--eval_script", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    if not args.eval_script.is_file():
        raise FileNotFoundError(args.eval_script)

    csv_files = list(args.root_dir.rglob("results.csv"))
    if not csv_files:
        print(f"No results.csv files found under {args.root_dir}")
        return

    rows: List[Dict[str, str]] = []
    for csv_path in sorted(csv_files):
        row = {"filename": str(csv_path.relative_to(args.root_dir))}
        success = True
        for k in (5, 10):
            try:
                m = run_eval(args.eval_script, csv_path, k)
            except Exception as e:
                success = False
                print(f"[WARN] {csv_path} (k={k}) skipped: {e}")
                break
            row[f"avg_precision{k}"] = f"{m['avg_precision']:.6f}"
            row[f"hit{k}"]           = f"{m['hit']:.6f}"

        if success:
            rows.append(row)
            print(f"[OK] {row['filename']}  k=5,10 collected")

    if rows:
        write_rows(args.out_csv, rows)
        print(f"\n✔ Wrote {len(rows)} rows to {args.out_csv}")
    else:
        print("No metrics collected.")


if __name__ == "__main__":
    main()
