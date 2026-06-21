import argparse
import pandas as pd
import numpy as np
import math

from ast import literal_eval
import metrics.metrics as metrics

def removeDuplicates(topk_files: list) -> list:
    # Deduplicate topk files list (predicted) as many chunks can match in the same file
    # Remove duplicate predictions while preserving order.
    if not isinstance(topk_files, list):
        return []

    seen = set()
    deduped_predictions = []
    for pred in topk_files:
        if pred not in seen:
            seen.add(pred)
            deduped_predictions.append(pred)
    return deduped_predictions

def calculateResults(column: str,
                     gt_column: str,
                     df: pd.DataFrame,
                     k_col: str = "top_k",
                     column_suffix: str = "", max_k = 0):
    """Compute all metrics, using df[k_col] as the *per‑row* k."""
    sf_tp = 0
    sf_fp = 0
    sf_fn = 0
    sf_mrr = 0
    sf_k = 0
    mf_tp = 0
    mf_fp = 0
    mf_fn = 0
    mf_mrr = 0
    mf_k = 0

    # ----- helpers ---------------------------------------------------------
    def k(row):
        if max_k > 0:
            return max_k
        return int(row[k_col])          # convenience, k varies by row

    def preds(row):
        """
        Deduplicate and truncate to the row-specific k.
        Guarantees len(preds(row)) == k(row)  (unless the list is shorter).
        """
        dedup = removeDuplicates(row[column])
        k_eff = min(k(row), len(dedup))
        return dedup[: k_eff]

    # single‑row metrics ----------------------------------------------------
    df[f'precision@2{column_suffix}'] = df.apply(
        lambda r: metrics.compute_precision_at_2_single(r[gt_column],
                                                        preds(r)),
        axis=1
    )
    df[f'precision@k{column_suffix}'] = df.apply(
        lambda r: metrics.compute_precision_at_k(r[gt_column],
                                                 preds(r),
                                                 k=k(r)),
        axis=1
    )
    df[f'recall@1{column_suffix}'] = df.apply(
        lambda r: metrics.compute_recall_at_1_single(r[gt_column],
                                                     preds(r)),
        axis=1
    )
    df[f'recall@2{column_suffix}'] = df.apply(
        lambda r: metrics.compute_recall_at_2_single(r[gt_column],
                                                     preds(r)),
        axis=1
    )
    df[f'MAP{column_suffix}'] = df.apply(
        lambda r: metrics.compute_average_precision(r[gt_column],
                                                    preds(r)),
        axis=1
    )
    df[f'recall@k{column_suffix}'] = df.apply(
        lambda r: metrics.compute_recall_at_k(r[gt_column],
                                              preds(r),
                                              k=k(r)),
        axis=1
    )
    df[f'f1@k{column_suffix}'] = df.apply(
        lambda r: metrics.compute_f1_at_k(r[f'recall@k{column_suffix}'],
                                          r[f'precision@k{column_suffix}'],
                                          k=k(r)),
        axis=1
    )
    df[f'hit_rate@k{column_suffix}'] = df.apply(
        lambda r: metrics.hit_rate_at_k(r[gt_column],
                                        preds(r),
                                        k=k(r)),
        axis=1
    )
    df[f'all_files_predicted{column_suffix}'] = df.apply(
        lambda r: metrics.all_files_in_predicted(r[gt_column],
                                                 preds(r)),
        axis=1
    )
    df['MRR'] = df.apply(
        lambda r: metrics.mean_reciprocal_rank(r[gt_column],
                                               preds(r)),
        axis=1
    )

    sf_subset = df[df['recall@1'+column_suffix].notna()]
    for _, row in sf_subset.iterrows():
        gt_set = set(row[gt_column])
        pred_set = set(preds(row))
        
        sf_tp += len(gt_set & pred_set)
        sf_fp += len(pred_set - gt_set)
        sf_fn += len(gt_set - pred_set)
        sf_k += len(pred_set)
    
    mf_subset = df[df['recall@2'+column_suffix].notna()]
    for _, row in mf_subset.iterrows():
        gt_set = set(row[gt_column])
        pred_set = set(preds(row))
        
        mf_tp += len(gt_set & pred_set)
        mf_fp += len(pred_set - gt_set)
        mf_fn += len(gt_set - pred_set)
        mf_k += len(pred_set)

    sf_average_k = sf_k / len(sf_subset) if len(sf_subset) > 0 else 0
    mf_average_k = mf_k / len(mf_subset) if len(mf_subset) > 0 else 0
    r1_count = df['recall@1'+column_suffix].dropna().count()
    print(f"There are {r1_count} elements with only one ground truth file")
    r2_count = df['recall@2'+column_suffix].dropna().count()
    print(f"There are {r2_count} elements with 2 or more ground truth files")
    
    # Single-file benchmark
    sf_r1 = sf_subset['recall@1'+column_suffix].mean()
    sf_pk = sf_subset['precision@k'+column_suffix].mean()
    sf_ap = sf_subset['MAP'+column_suffix].dropna().mean()
    sf_r5 = sf_subset['recall@k'+column_suffix].dropna().mean()
    sf_f1_at_5 = sf_subset['f1@k'+column_suffix].dropna().mean()
    sf_hit_at_5 = sf_subset['hit_rate@k'+column_suffix].dropna().mean()
    sf_all_files_pred = sf_subset['all_files_predicted'+column_suffix].dropna().mean()
    sf_mrr = sf_subset['MRR'].dropna().mean()

    # Multiple-file benchmark
    mf_p2 = mf_subset['precision@2'+column_suffix].dropna().mean()
    mf_pk = mf_subset['precision@k'+column_suffix].dropna().mean()
    mf_r2 = mf_subset['recall@2'+column_suffix].dropna().mean()
    mf_ap = mf_subset['MAP'+column_suffix].dropna().mean()
    mf_r5 = mf_subset['recall@k'+column_suffix].dropna().mean()
    mf_f1_at_5 = mf_subset['f1@k'+column_suffix].dropna().mean()
    mf_hit_at_5 = mf_subset['hit_rate@k'+column_suffix].dropna().mean()
    mf_all_files_pred = mf_subset['all_files_predicted'+column_suffix].dropna().mean()
    mf_mrr = mf_subset['MRR'].dropna().mean()
    # Overall hit@k and MAP
    all_hitk = df['hit_rate@k'+column_suffix].dropna().mean()
    all_map = df['MAP'+column_suffix].dropna().mean()

    # Compute overall F1 score for single-file
    if (sf_tp + sf_fp + sf_fn) > 0 and sf_tp + sf_fn > 0 and sf_tp + sf_fp > 0:
        overall_sf_precision = sf_tp / (sf_tp + sf_fp)
        overall_sf_recall = sf_tp / (sf_tp + sf_fn)

        if overall_sf_precision + overall_sf_recall > 0:
            overall_sf_f1 = 2 * (overall_sf_precision * overall_sf_recall) / (overall_sf_precision + overall_sf_recall)
        else:
            overall_sf_f1 = 0.0
    else:
        overall_sf_f1 = np.nan


    # Compute overall F1 score for multi-file
    if (mf_tp + mf_fp + mf_fn) > 0 and mf_tp + mf_fn > 0 and mf_tp + mf_fp > 0:
        overall_mf_precision = mf_tp / (mf_tp + mf_fp)
        overall_mf_recall = mf_tp / (mf_tp + mf_fn)

        if overall_mf_precision + overall_mf_recall > 0:
            overall_mf_f1 = 2 * (overall_mf_precision * overall_mf_recall) / (overall_mf_precision + overall_mf_recall)
        else:
            overall_mf_f1 = 0.0
    else:
        overall_mf_f1 = np.nan

    print("---- Single file bugs ----")
    print(f"Overall Recall@1{column_suffix}", sf_r1)
    print(f"Overall Recall@k{column_suffix}", sf_r5)
    print(f"Overall Precision@k{column_suffix}:", sf_pk)
    print(f"Overall Average Precision{column_suffix}", sf_ap)
    print(f'Overall F1@k{column_suffix}', sf_f1_at_5)
    print(f'Overall F1 Score{column_suffix}', overall_sf_f1)
    print(f'Overall Hit Rate@k score{column_suffix}', sf_hit_at_5)
    print(f'Overall all files in predicted{column_suffix}', sf_all_files_pred)
    print(f'MRR: {sf_mrr}')
    print("Overall Average k:", sf_average_k)

    print("---- Multi-file bugs ----")
    print(f"Overall Precision@2{column_suffix}", mf_p2)
    print(f"Overall Precision@k{column_suffix}", mf_pk)
    print(f"Overall Recall@2{column_suffix}", mf_r2) 
    print(f"Overall Recall@k{column_suffix}", mf_r5)
    print(f"Overall Average Precision{column_suffix}", mf_ap)
    print(f'Overall F1@k{column_suffix}', mf_f1_at_5)
    print(f'Overall F1 Score{column_suffix}', overall_mf_f1)
    print(f'Overall Hit Rate@k score{column_suffix}', mf_hit_at_5)
    print(f'Overall all files in predicted{column_suffix}', mf_all_files_pred)
    print(f'MRR: {mf_mrr}')
    print("Overall Average k:", mf_average_k)


    print(f"------- Overall Hit@k and MAP {column_suffix}")
    print(f"Overall hit@k {column_suffix}: {all_hitk}")
    print(f"Overall MAP {column_suffix}: {all_map}")

def computeMetrics(outfile: str, percent: int, max_k: int):
    # Note: uncomment all files if using dependencies
    data = pd.read_csv(outfile,
                       converters={
                        "expected_files": literal_eval,
                        "changed_files": literal_eval,
                        "ranked_files":   literal_eval,   # <-- guarantees each cell is a list
                        "all_files": literal_eval,
                        "baseline_all_files": literal_eval,
    },)
    df = pd.DataFrame(data)

    # ---- new: per‑row top‑k ---------------------------------------------
    if percent > 0:
        df['top_k'] = df['topk_files'].apply(
           lambda lst: max(1, math.ceil(percent/100 * len(removeDuplicates(lst))))
        )
    else:
        # use the whole list length for each row
        df['top_k'] = df['ranked_files'].apply(
            lambda lst: len(removeDuplicates(lst)))

    if not "expected_files" in df:
        gt_column = "changed_files"
    else:
        gt_column = "expected_files"

    # ---- compute metrics -------------------------------------------------
    calculateResults('ranked_files', gt_column, df, k_col='top_k', max_k=max_k)

    # Note: uncomment all files if using dependencies
    #df['top_k'] = df['all_files'].apply(len)
    #calculateResults('all_files', df, k_col='top_k', column_suffix='_w_dep', max_k=max_k)

    #df['baseline_top_k'] = df['baseline_all_files'].apply(len)
    #calculateResults('baseline_all_files', df, k_col='baseline_top_k',
    #                 column_suffix='_baseline', max_k=max_k)
    # ---- persist ---------------------------------------------------------
    df.to_csv(outfile, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True)
    parser.add_argument("--percent", type=int, default=0)
    parser.add_argument("--max_k", type=int, default=0)
    args = parser.parse_args()
    computeMetrics(args.input, args.percent, args.max_k)


if __name__ == "__main__":
    main()
