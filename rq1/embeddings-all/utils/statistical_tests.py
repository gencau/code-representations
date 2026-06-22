
import argparse
import os
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import binom, wilcoxon

from metrics import metrics

index_column = "index"
found_files_column = "topk_files"
ground_truth_column = "expected_files"

def removeDuplicates(topk_files: list) -> list:
    # Deduplicate topk files list (predicted) as many chunks can match in the same file
    # Remove duplicate predictions while preserving order.
    if not isinstance(topk_files, list):
        return []

    seen = set()
    deduped_predictions = []
    for pred in topk_files:
        if isinstance(pred, dict):
            continue
        if pred not in seen:
            seen.add(pred)
            deduped_predictions.append(pred)
    return deduped_predictions

def calculateResults(column: str,
                     df: pd.DataFrame,
                     k_col: str = "top_k",
                     column_suffix: str = "", 
                     column_prefix: str = ""):
    """Compute all metrics, using df[k_col] as the *per-row* k."""

    # ----- helpers ---------------------------------------------------------
    def k(row):
        return int(row[k_col])          # convenience, k varies by row

    def preds(row):
        """
        Deduplicate and truncate to the row-specific k.
        Guarantees len(preds(row)) == k(row)  (unless the list is shorter).
        """
        dedup = removeDuplicates(row[column])
        return dedup[: k(row)]

    df[f'{column_prefix}MAP{column_suffix}'] = df.apply(
        lambda r: metrics.compute_average_precision(r[ground_truth_column],
                                                    preds(r)),
        axis=1
    )
    df[f'{column_prefix}hit_rate@k{column_suffix}'] = df.apply(
        lambda r: metrics.hit_rate_at_k(r[ground_truth_column],
                                        preds(r),
                                        k=k(r)),
        axis=1
    )

def delta_magnitude(delta: float) -> str:
    if np.isnan(delta):
        return "NA"
    a = abs(delta)
    if a < 0.10:
        return "negligible"
    elif a < 0.30:
        return "small"
    elif a < 0.50:
        return "medium"
    else:
        return "large"

# ----------------- IO & Testing -----------------

def load_exp_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, converters={
        ground_truth_column: literal_eval,
        found_files_column: literal_eval
    })
    if index_column not in df.columns:
        raise ValueError(f"{path} missing 'index' column")
    return df

def compute_map_hit(df: pd.DataFrame, k: int) -> pd.DataFrame:
    
    # Set per-row k
    df = df.copy()
    df["top_k"] = k

    calculateResults(found_files_column, df, k_col='top_k')
    # Return just the columns we need
    out = df[[index_column, "MAP", "hit_rate@k"]].copy()
    out = out.rename(columns={"hit_rate@k":"Hit"})
    return out

# ----------------- Pairwise testing -----------------

def mcnemar_for_hit(sa: pd.Series, sb: pd.Series):
    """Run McNemar on paired binary series A (sa) vs B (sb)."""
    both = pd.DataFrame({"A": sa.astype(int), "B": sb.astype(int)}).dropna()
    A = both["A"].astype(int)
    B = both["B"].astype(int)
    # Full 2x2 matched-pairs table components
    a11 = int(((A == 1) & (B == 1)).sum())  # both hit
    a00 = int(((A == 0) & (B == 0)).sum())  # both miss
    b01 = int(((A == 0) & (B == 1)).sum())  # A miss, B hit
    b10 = int(((A == 1) & (B == 0)).sum())  # A hit,  B miss
    table = [[a11, b10], [b01, a00]]

    n_disc = b01 + b10

    # Calculate the exact p-value.
    i: int = table[0][1]
    n: int = table[1][0] + table[0][1]
    i_n: np.ArrayLike = np.arange(i + 1, n + 1)

    p_value_exact: float = 2 * (1 - np.sum(binom.pmf(i_n, n, 0.5)))

    if n_disc < 25:
        p = p_value_exact
    else:
        mid_p_value: float = p_value_exact - binom.pmf(table[0][1], n, 0.5)
        p = mid_p_value

    p = float(p)
    # Extra interpretable stats for paired binary:
    net_gain = b01 - b10
    cohens_g = (net_gain / n_disc) if n_disc > 0 else 0.0  # effect on discordants
    odds_ratio = ((b01 + 0.5) / (b10 + 0.5)) if n_disc > 0 else (float('inf') if b01 > 0 else np.nan)
    return p, b01, b10, net_gain, cohens_g, odds_ratio

def wilcoxon_effect(x, y):
    """
    Paired Wilcoxon signed-rank test.
    Returns: p_value, effect_size_r, n_used
    effect_size_r = |z| / sqrt(n)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    d = x - y
    mask = ~np.isnan(d)
    d = d[mask]

    # Wilcoxon drops zero diffs via zero_method='wilcox'
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return np.nan, np.nan, 0

    # scipy can take the two paired samples directly
    stat, p = wilcoxon(x[mask], y[mask], zero_method="wilcox")

    # Convert W -> z (normal approx), then r = |z| / sqrt(n)
    # W here is the sum of signed ranks for positive diffs (per SciPy)
    # Use standard normal approx with continuity correction omitted for simplicity.
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (stat - mu) / sigma if sigma > 0 else 0.0
    r = abs(z) / np.sqrt(n)

    return float(p), float(r)

def pairwise_per_k(per_exp_data: dict, metric: str):
    labels = sorted(per_exp_data.keys())
    rows = []
    p_pairs = []
    pmat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    dmat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    def series(label):
        return per_exp_data[label].set_index(index_column)[metric]

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = labels[i], labels[j]
            sa, sb = series(a), series(b)
            common = sa.index.intersection(sb.index)
            xa = sa.loc[common].values
            yb = sb.loc[common].values

            n_common = len(common)
            mean_a = float(np.nanmean(xa)) if n_common else np.nan
            mean_b = float(np.nanmean(yb)) if n_common else np.nan
            diff_mean = (mean_a - mean_b) if n_common else np.nan

            if metric == "MAP":
                p, delta = wilcoxon_effect(xa, yb)
                U = np.nan  # not applicable for paired MAP
                b01 = b10 = net_gain = cohens_g = odds_ratio = np.nan
            else:  # Hit -> McNemar
                if n_common == 0:
                    U = np.nan
                    p = np.nan
                    delta = np.nan
                    b01 = b10 = net_gain = cohens_g = odds_ratio = np.nan
                else:
                    # McNemar uses paired 0/1 series:
                    sA = sa.loc[common].astype(int)
                    sB = sb.loc[common].astype(int)
                    p, b01, b10, net_gain, cohens_g, odds_ratio = mcnemar_for_hit(sA, sB)
                    U = np.nan  # not used for Hit
                    delta = np.nan

            rows.append({
                "A": a, "B": b, "metric": metric,
                "n_common": n_common,
                "mean_A": mean_a, "mean_B": mean_b, "diff_mean": diff_mean,
                "U": U, "p_raw": p,
                "delta": delta, "delta_magnitude": delta_magnitude(delta),
                # McNemar-specific diagnostics (NaN for MAP):
                "b01_A0_B1": b01, "b10_A1_B0": b10,
                "net_gain_B_minus_A": net_gain,
                "cohens_g_on_discordants": cohens_g,
                "discordant_odds_ratio": odds_ratio,
            })
            p_pairs.append(((a,b), float(p) if not np.isnan(p) else np.nan))
            pmat.loc[a,b] = pmat.loc[b,a] = p
            dmat.loc[a,b] = delta
            dmat.loc[b,a] = -delta if not np.isnan(delta) else np.nan

    np.fill_diagonal(pmat.values, 0.0)
    np.fill_diagonal(dmat.values, 0.0)

    return pd.DataFrame(rows), pmat, dmat

def main():
    ap = argparse.ArgumentParser(description="Pairwise sigtests: Wilcoxon for MAP, McNemar for Hit@k (paired binary).")
    ap.add_argument("--exp", action="append", required=True, help="LABEL=PATH/to/results.csv  (repeatable)")
    ap.add_argument("--k", type=int, nargs="+", default=[1,5,10,20], help="Top-k values to compare.")
    ap.add_argument("--tag", type=str, default="ALL", help="Tag used in output filenames, e.g., LCA or SWE.")
    ap.add_argument("--outdir", type=str, default="sigtests_outputs_same_metrics")
    ap.add_argument("--type", type=str, choices=["bm25", "embedding"], default="embedding", help="Type of retrieval used.")
    args = ap.parse_args()

    # Parse experiments
    exps = {}
    for spec in args.exp:
        if "=" not in spec:
            raise ValueError(f"--exp must be LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        exps[label.strip()] = path.strip()

    os.makedirs(args.outdir, exist_ok=True)

    if args.type == "bm25":
        print("Using BM25-based retrieval results for statistical tests.")
        global index_column, found_files_column, ground_truth_column
        index_column = "id"
        found_files_column = "final_files"
        ground_truth_column = "changed_files"

    for k in args.k:
        # Load and compute per-item MAP & Hit with user's calculations
        per_exp = {}
        for label, path in exps.items():
            df = load_exp_results(path)
            vals = compute_map_hit(df, k)
            per_exp[label] = vals

        # Pairwise tests
        for metric in ["MAP", "Hit"]:
            df_summary, pmat, dmat = pairwise_per_k(per_exp, metric)
            base = f"{args.tag}_k{k}_{metric}"
            df_summary.sort_values(by=["p_raw","delta"], ascending=[True,False]).to_csv(
                os.path.join(args.outdir, f"summary_{base}.csv"), index=False
            )
            pmat.to_csv(os.path.join(args.outdir, f"pvalues_{base}.csv"))
            dmat.to_csv(os.path.join(args.outdir, f"delta_{base}.csv"))
            print(f"[k={k} {metric}] wrote: summary_{base}.csv, pvalues_{base}.csv, delta_{base}.csv")

if __name__ == "__main__":
    main()
