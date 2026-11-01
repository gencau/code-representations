import argparse
import ast
import contextlib
import io
import os
import numpy as np
import sys

import pandas as pd

# Shared metric implementations live in code-representations/common/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 6)))
# The bug_localization root, for the src.* imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from common.classification_metrics import compute_precision_at_2_single,  \
    compute_recall_at_1_single, compute_recall_at_2_single, compute_average_precision, hit_rate_at_k, \
    all_files_in_predicted, compute_precision_at_k, compute_recall_at_k, acc_at_k
from src.utils.git_utils import parse_changed_files_from_diff


# The ground truth comes from the results file itself (patch for SWE-bench, changed_files for LCA),
# so no Hydra config or data source is needed
def compute_metrics(results_csv_path: str, topk: int = 0, rerank: bool = False, write: bool = False) -> dict:
    df = pd.read_csv(results_csv_path)

    if rerank:
        print("****** Using RERANKED files")

    precision_at_2_list = []
    recall_at_1_list = []
    recall_at_2_list = []
    ap_list = []
    hit_at_k_list = []
    tcr_list = []
    precision_k_list = []
    hit_at_1_list = []
    hit_at_3_list = []
    recall_k_list = []
    acc_k_list = []

    #Overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0

    index = 0
    total = len(df)
    print(f"Processing {total} records...")

    #for dp, repo_content, changed_files in data_source:
    for _, row in df.iterrows():
        if index >= len(df):
            # in case we get an empty line at the end
            break

        if "patch" in df:
            changed_files = parse_changed_files_from_diff(row['patch'])
        else:
            changed_files = ast.literal_eval(row['changed_files'])#parse_changed_files_from_diff(row['diff'])

        # all generated files may contain invalid files
        # final files is cleaned up to contain only valid files
        if rerank:
            raw_string = row['reranked_files']
            expected_files = ast.literal_eval(row["reranked_files"])
        else:
            expected_files = ast.literal_eval(row['all_generated_files'])
            raw_string = row['final_files']
        
        # Remove the brackets and split by comma
        final_files = [item.strip().strip("'").strip('"') for item in raw_string.strip("[]").split(",") if item.strip()]
        
        # use the expected files as the predicted and validate the files
        if rerank:
            for f in expected_files:
                if "final_reranked_files" in df:
                    final_files = ast.literal_eval(row['final_reranked_files'])
                elif "cleaned_reranked_files" in df:
                    final_files = ast.literal_eval(row['cleaned_reranked_files'])


        gt_files = list(changed_files)
        predicted_list = list(expected_files) 
        # filter invalid predictions
        predicted_list = [p for p in predicted_list if isinstance(p, str)]
        
        seen = set()
        pred_no_dupes = []
        for p in predicted_list:
            if p not in seen:
                pred_no_dupes.append(p)
                seen.add(p)
        # Hit@1 and Hit@3 use the deduplicated ranking before the topk cut,
        # so they match the Hit@k of a run with topk=1 or topk=3
        hit_at_1 = hit_rate_at_k(gt_files, pred_no_dupes, 1)
        hit_at_3 = hit_rate_at_k(gt_files, pred_no_dupes, 3)

        if topk > 0:
            pred_no_dupes = pred_no_dupes[:topk]
            final_files = final_files[:topk]

        pred_set = set(pred_no_dupes)
        gt_set = set(gt_files)
        precision_at_2 = -1
        recall_at_1 = -1
        recall_at_2 = -1

        # Precision @ 1
        precision_at_2 = compute_precision_at_2_single(gt_files, pred_no_dupes)
        recall_at_1 = compute_recall_at_1_single(gt_files, pred_no_dupes)
        recall_at_2 = compute_recall_at_2_single(gt_files, pred_no_dupes)
        average_precision = compute_average_precision(gt_files, pred_no_dupes)
        precision_k = compute_precision_at_k(gt_files, predicted_list, len(pred_no_dupes))
        hit_at_k = hit_rate_at_k(gt_files, pred_no_dupes, len(pred_no_dupes))
        tcr = all_files_in_predicted(gt_files, pred_no_dupes)
        recall_k = compute_recall_at_k(gt_files, pred_no_dupes, len(pred_no_dupes))
        acc_k = acc_at_k(gt_files, pred_no_dupes, len(pred_no_dupes))

        # Update overall TP, FP, FN
        # For multi-label, compute TP, FP, FN
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision_at_2_list.append(precision_at_2)
        recall_at_1_list.append(recall_at_1)
        recall_at_2_list.append(recall_at_2)
        ap_list.append(average_precision)
        precision_k_list.append(precision_k)
        hit_at_k_list.append(hit_at_k)
        tcr_list.append(tcr)
        hit_at_1_list.append(hit_at_1)
        hit_at_3_list.append(hit_at_3)
        recall_k_list.append(recall_k)
        acc_k_list.append(acc_k)

        print('Precision at 2:', precision_at_2)
        print('Recall@1:', recall_at_1)
        print('Recall@2:', recall_at_2)
        print('Average precision:', average_precision)
        print('Ground truth:', changed_files)
        print('Valid predictions:', final_files)
        print('Predicted:', expected_files)
        print('Common:', set(final_files).intersection(changed_files))
        index += 1

    # Add metrics as new columns to the DataFrame
    df['Precision@2'] = precision_at_2_list
    df['Recall@1'] = recall_at_1_list
    df['Recall@2'] = recall_at_2_list
    df['AP'] = ap_list
    df['precision_k'] = precision_k_list
    df['hit@k'] = hit_at_k_list
    df['tcr'] = tcr_list
    df['hit@1'] = hit_at_1_list
    df['hit@3'] = hit_at_3_list
    df['recall@k'] = recall_k_list
    df['acc@k'] = acc_k_list

    # Drop NaN values to consider only relevant bugs
    overall_p2 = df['Precision@2'].dropna().mean()
    overall_r1 = df['Recall@1'].dropna().mean()
    overall_r2 = df['Recall@2'].dropna().mean()
    overall_ap = df['AP'].dropna().mean()
    overall_precision_k = df['precision_k'].dropna().mean()
    overall_hit_k = df['hit@k'].dropna().mean()
    overall_tcr = df['tcr'].dropna().mean()

    # Compute overall F1 score
    if (total_tp + total_fp + total_fn) > 0:
        overall_precision = total_tp / (total_tp + total_fp)
        overall_recall = total_tp / (total_tp + total_fn)
        if (overall_precision + overall_recall) > 0:
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        else:
            overall_f1 = 0.0
    else:
        overall_f1 = np.nan  # Undefined if no predictions and no ground truths

    print("Overall Precision@2 (multi-file bugs):", overall_p2)
    print("Overall Recall@1 (single-file bugs):", overall_r1)
    print("Overall Recall@2 (multi-file bugs):", overall_r2) 
    print("Overall Average Precision:", overall_ap)
    print('Overall F1 score:', overall_f1)
    print("Overall precision@k: ", overall_precision_k)
    print("Overall Hit@k: ", overall_hit_k)
    print("Overall TCR: ", overall_tcr)
    print("Overall Hit@1: ", df['hit@1'].dropna().mean())
    print("Overall Hit@3: ", df['hit@3'].dropna().mean())
    print("Overall Recall@k: ", df['recall@k'].dropna().mean())
    print("Overall Acc@k: ", df['acc@k'].dropna().mean())

    # Calculate AP for single and multi file tasks separately
    single_file_ap = df[df['changed_files_count'] == 1]['AP'].mean()
    multi_file_ap = df[df['changed_files_count'] > 1]['AP'].mean()
    hit_k_single = df[df['changed_files_count'] == 1]['hit@k'].mean()
    hit_k_multi = df[df['changed_files_count'] > 1]['hit@k'].mean()
    print(f"Single-file Hit@k: {hit_k_single}")
    print(f"Multi-file hit@k: {hit_k_multi}")
    print(f"Multi-file Recall@k: {df[df['changed_files_count'] > 1]['recall@k'].mean()}")
    print(f"Multi-file Acc@k: {df[df['changed_files_count'] > 1]['acc@k'].mean()}")

    print(f"Single file MAP: {single_file_ap}")
    print(f"Multi-file MAP: {multi_file_ap}")
    df['single_file_AP'] = single_file_ap
    df['multi_file_AP'] = multi_file_ap

    if write:
        df.to_csv(results_csv_path, index=False)

    return {"MAP": overall_ap, "Hit@1": df['hit@1'].dropna().mean(), "Hit@3": df['hit@3'].dropna().mean(),
            "Hit@k": overall_hit_k}

# Call like this: python compute_metrics_llm.py --topk 5 <run folder or results.csv> [...]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score LLM bug localization results.csv files")
    parser.add_argument("paths", nargs="+", help="results.csv files, or run folders that contain one")
    parser.add_argument("--topk", type=int, default=5, help="cut each ranking to its top k files (0: no cut)")
    parser.add_argument("--rerank", action="store_true", help="score reranked_files instead of all_generated_files")
    parser.add_argument("--write", action="store_true", help="write the metric columns back into results.csv")
    parser.add_argument("--quiet", action="store_true", help="print only the summary table")
    args = parser.parse_args()

    scores = []
    for path in args.paths:
        csv_path = os.path.join(path, "results.csv") if os.path.isdir(path) else path
        with contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext():
            scores.append((csv_path, compute_metrics(csv_path, args.topk, args.rerank, args.write)))

    k = args.topk if args.topk > 0 else "k"
    print(f"{f'MAP@{k}':>8} {'Hit@1':>6} {'Hit@3':>6} {f'Hit@{k}':>6}  run")
    for csv_path, m in scores:
        print(f"{m['MAP']:8.3f} {m['Hit@1']:6.3f} {m['Hit@3']:6.3f} {m['Hit@k']:6.3f}  {csv_path}")
