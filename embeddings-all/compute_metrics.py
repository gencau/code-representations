import argparse
import pandas as pd
import numpy as np

from ast import literal_eval
from metrics import metrics


def computeMetrics(outfile):
    with open(outfile, mode='r') as file:
        data = pd.read_csv(outfile)
        df = pd.DataFrame(data)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Convert string columns to lists
        df['expected_files'] = df['expected_files'].apply(literal_eval)
        df['topk_files'] = df['topk_files'].apply(literal_eval)

        # Compute metrics for each row using apply()
        df['precision@2'] = df.apply(
            lambda row: metrics.compute_precision_at_2_single(row['expected_files'], row['topk_files']),
            axis=1
        )
        df['precision@5'] = df.apply(
            lambda row: metrics.compute_precision_at_k(row['expected_files'], row['topk_files'], k=5),
            axis=1
        )
        df['recall@1'] = df.apply(
            lambda row: metrics.compute_recall_at_1_single(row['expected_files'], row['topk_files']),
            axis=1
        )
        df['recall@2'] = df.apply(
            lambda row: metrics.compute_recall_at_2_single(row['expected_files'], row['topk_files']),
            axis=1
        )
        df['MAP'] = df.apply(
            lambda row: metrics.compute_average_precision(row['expected_files'], row['topk_files']),
            axis=1
        )
        df['recall@5'] = df.apply(
            lambda row: metrics.compute_recall_at_k(row['expected_files'], row['topk_files'], k=len(row['topk_files'])),
            axis=1
        )
        df['f1@5'] = df.apply(
            lambda row: metrics.compute_f1_at_k(row['recall@5'], row['precision@5'], k=len(row['topk_files'])),
            axis=1
        )
        df['hit_rate@5'] = df.apply(
            lambda row: metrics.hit_rate_at_k(row['expected_files'], row['topk_files'], k=5),
            axis=1
        )

        for _, row in df.iterrows():
            gt_set = set(row['expected_files'])
            pred_set = set(row['topk_files'])
            
            total_tp += len(gt_set & pred_set)
            total_fp += len(pred_set - gt_set)
            total_fn += len(gt_set - gt_set)
        
        overall_p2 = df['precision@2'].dropna().mean()
        overall_r1 = df['recall@1'].dropna().mean()
        overall_r2 = df['recall@2'].dropna().mean()
        overall_ap = df['MAP'].dropna().mean()
        overall_r5 = df['recall@5'].dropna().mean()
        overall_f1_at_5 = df['f1@5'].dropna().mean()
        overall_hit_at_5 = df['hit_rate@5'].dropna().mean()
        overall_p5 = df['precision@5'].dropna().mean()

        # Compute overall F1 score
        if (total_tp + total_fp + total_fn) > 0:
            overall_precision = 0.0
            overall_recall = 0.0
            if (total_tp + total_fp) > 0:
                overall_precision = total_tp / (total_tp + total_fp)
            if (total_tp + total_fn) > 0:
                overall_recall = total_tp / (total_tp + total_fn)

            if (overall_precision + overall_recall) > 0:
                overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
            else:
                overall_f1 = 0.0
        else:
            overall_f1 = np.nan

        print("Overall Precision@2 (single-file bugs):", overall_p2)
        print("Overall Precision@5:", overall_p5)
        print("Overall Recall@1 (single-file bugs):", overall_r1)
        print("Overall Recall@2 (multi-file bugs):", overall_r2) 
        print("Overall Average Precision:", overall_ap)
        print('Overall F1 score:', overall_f1)
        print('Overall Recall@5 score:', overall_r5)
        print('Overall F1@5 score:', overall_f1_at_5)
        print('Overall Hit Rate@5 score:', overall_hit_at_5)
        df.to_csv(outfile, index=False)

def main():
    parser = argparse.ArgumentParser(description="Process metrics on results file.")

    parser.add_argument("--input", type=str, required=True, help='Path to input file')

    args = parser.parse_args()
    computeMetrics(args.input)

if __name__ == "__main__":
    main()