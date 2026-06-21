import ast
import os
import numpy as np
import sys

import hydra
import pandas as pd
from dotenv import load_dotenv

from src.baselines.configs.baseline_configs import BaselineConfig
from src.baselines.data_sources.base_data_source import BaseDataSource
from src.baselines.metrics.classification_metrics import compute_precision_at_2_single,  \
    compute_recall_at_1_single, compute_recall_at_2_single, compute_average_precision, hit_rate_at_k, \
    all_files_in_predicted, compute_precision_at_k
from src.utils.git_utils import parse_changed_files_from_diff


@hydra.main(version_base="1.1", config_path="../../../configs/baselines",  config_name="qwen-agent")
def main(cfg: BaselineConfig) -> None:
    print("Final configuration: ", cfg)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Insert the project root at the beginning of sys.path so it has priority
    sys.path.insert(0, project_root)

    data_source: BaseDataSource = hydra.utils.instantiate(cfg.data_source)
    results_path = os.path.join(cfg.output_path, cfg.name) 
    os.makedirs(results_path, exist_ok=True)
    results_csv_path = os.path.join(results_path, "results.csv")
    topk = cfg.get("topk", 0)
    df = pd.read_csv(results_csv_path)

    rerank = False
    if (cfg.backbone.experiment == "rerank"):
        rerank = True
        print("****** Using RERANKED files")

    precision_at_2_list = []
    recall_at_1_list = []
    recall_at_2_list = []
    ap_list = []
    hit_at_k_list = []
    tcr_list = []
    precision_k_list = []

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

    # Calculate AP for single and multi file tasks separately
    single_file_ap = df[df['changed_files_count'] == 1]['AP'].mean()
    multi_file_ap = df[df['changed_files_count'] > 1]['AP'].mean()
    hit_k_single = df[df['changed_files_count'] == 1]['hit@k'].mean()
    hit_k_multi = df[df['changed_files_count'] > 1]['hit@k'].mean()
    print(f"Single-file Hit@k: {hit_k_single}")
    print(f"Multi-file hit@k: {hit_k_multi}")

    print(f"Single file MAP: {single_file_ap}")
    print(f"Multi-file MAP: {multi_file_ap}")
    df['single_file_AP'] = single_file_ap
    df['multi_file_AP'] = multi_file_ap

    df.to_csv(results_csv_path, index=False)

# Call like this: python compute_metrics.py +output_path="/Users/gen/workspace/lca-baselines/bug_localization/output" +name="openai_chat_gpt-3.5-turbo-1106" 
if __name__ == '__main__':
    load_dotenv()
    main()
