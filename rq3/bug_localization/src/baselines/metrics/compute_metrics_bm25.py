import argparse
import pandas as pd
import numpy as np

from ast import literal_eval
import src.baselines.metrics.classification_metrics as metrics

def removeDuplicates(topk_files: list) -> list:
    # Deduplicate topk files list (predicted) as many chunks can match in the same file
    # Remove duplicate predictions while preserving order.
    if not isinstance(topk_files, list):
        return []
    
    seen = set()
    deduped_predictions = []
    print(topk_files)
    for pred in topk_files:
        if pred not in seen:
            seen.add(pred)
            deduped_predictions.append(pred)
    return deduped_predictions

def computeMetrics(outfile):
    with open(outfile, mode='r') as file:
        data = pd.read_csv(outfile)
        df = pd.DataFrame(data)

        sf_tp = 0
        sf_fp = 0
        sf_fn = 0
        mf_tp = 0
        mf_fp = 0
        mf_fn = 0

        top_k = len(df['final_files'])
        # Convert string columns to lists
        df['changed_files'] = df['changed_files'].apply(literal_eval)
        df['final_files'] = df['final_files'].apply(literal_eval)

        # Compute metrics for each row using apply()
        df['precision@2'] = df.apply(
            lambda row: metrics.compute_precision_at_2_single(row['changed_files'], removeDuplicates(row['final_files'])),
            axis=1
        )
        df['precision@k'] = df.apply(
            lambda row: metrics.compute_precision_at_k(row['changed_files'], removeDuplicates(row['final_files']), k=top_k),
            axis=1
        )
        df['recall@1'] = df.apply(
            lambda row: metrics.compute_recall_at_1_single(row['changed_files'], removeDuplicates(row['final_files'])),
            axis=1
        )
        df['recall@2'] = df.apply(
            lambda row: metrics.compute_recall_at_2_single(row['changed_files'], removeDuplicates(row['final_files'])),
            axis=1
        )
        df['MAP'] = df.apply(
            lambda row: metrics.compute_average_precision(row['changed_files'], removeDuplicates(row['final_files'])),
            axis=1
        )
        df['recall@k'] = df.apply(
            lambda row: metrics.compute_recall_at_k(row['changed_files'], removeDuplicates(row['final_files']), k=top_k),
            axis=1
        )
        df['f1@k'] = df.apply(
            lambda row: metrics.compute_f1_at_k(row['recall@k'], row['precision@k'], k=top_k),
            axis=1
        )
        df['hit_rate@k'] = df.apply(
            lambda row: metrics.hit_rate_at_k(row['changed_files'], removeDuplicates(row['final_files']), k=len(removeDuplicates(row['final_files']))),
            axis=1
        )
        df['all_files_predicted'] = df.apply(lambda row: metrics.all_files_in_predicted(row['changed_files'], removeDuplicates(row['final_files'])),
            axis=1
        )
        df['MRR'] = df.apply(lambda row: metrics.mean_reciprocal_rank(row['changed_files'], removeDuplicates(row['final_files'])), 
            axis=1
        )

        sf_subset = df[df['recall@1'].notna()]
        for _, row in sf_subset.iterrows():
            gt_set = set(row['changed_files'])
            pred_set = set(row['final_files'])
            
            sf_tp += len(gt_set & pred_set)
            sf_fp += len(pred_set - gt_set)
            sf_fn += len(gt_set - pred_set)
        
        mf_subset = df[df['recall@2'].notna()]
        for _, row in mf_subset.iterrows():
            gt_set = set(row['changed_files'])
            pred_set = set(row['final_files'])
            
            mf_tp += len(gt_set & pred_set)
            mf_fp += len(pred_set - gt_set)
            mf_fn += len(gt_set - pred_set)

        r1_count = df['recall@1'].dropna().count()
        print(f"There are {r1_count} elements with only one ground truth file")
        r2_count = df['recall@2'].dropna().count()
        print(f"There are {r2_count} elements with 2 or more ground truth files")

        
        # Single-file benchmark
        sf_r1 = sf_subset['recall@1'].mean()
        sf_pk = sf_subset['precision@k'].mean()
        sf_ap = sf_subset['MAP'].dropna().mean()
        sf_r5 = sf_subset['recall@k'].dropna().mean()
        sf_f1_at_5 = sf_subset['f1@k'].dropna().mean()
        sf_hit_at_5 = sf_subset['hit_rate@k'].dropna().mean()
        sf_all_files_pred = sf_subset['all_files_predicted'].dropna().mean()
        sf_mrr = sf_subset['MRR'].dropna().mean()

        # Multiple-file benchmark
        mf_p2 = mf_subset['precision@2'].dropna().mean()
        mf_pk = mf_subset['precision@k'].dropna().mean()
        mf_r2 = mf_subset['recall@2'].dropna().mean()
        mf_ap = mf_subset['MAP'].dropna().mean()
        mf_r5 = mf_subset['recall@k'].dropna().mean()
        mf_f1_at_5 = mf_subset['f1@k'].dropna().mean()
        mf_hit_at_5 = mf_subset['hit_rate@k'].dropna().mean()
        mf_all_files_pred = mf_subset['all_files_predicted'].dropna().mean()
        mf_mrr = mf_subset['MRR'].dropna().mean()

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
        print("Overall Recall@1:", sf_r1)
        print("Overall Recall@k:", sf_r5)
        print("Overall Precision@k:", sf_pk)
        print("Overall Average Precision:", sf_ap)
        print('Overall F1@k:', sf_f1_at_5)
        print('Overall Hit Rate@k score:', sf_hit_at_5)
        print('Overall all files in predicted:', sf_all_files_pred)
        print('Mean Reciprocal Rank:', sf_mrr)

        print("---- Multi-file bugs ----")
        print("Overall Precision@2 :", mf_p2)
        print("Overall Precision@k:", mf_pk)
        print("Overall Recall@2:", mf_r2) 
        print("Overall Recall@k:", mf_r5)
        print("Overall Average Precision:", mf_ap)
        print('Overall F1@k:', mf_f1_at_5)
        print('Overall Hit Rate@k score:', mf_hit_at_5)
        print('Overall all files in predicted:', mf_all_files_pred)
        print('Mean Reciprocal Rank:', mf_mrr)
        df.to_csv(outfile, index=False)

def main():
    parser = argparse.ArgumentParser(description="Process metrics on results file.")

    parser.add_argument("--input", type=str, required=True, help='Path to input file')

    args = parser.parse_args()
    computeMetrics(args.input)

if __name__ == "__main__":
    main()
