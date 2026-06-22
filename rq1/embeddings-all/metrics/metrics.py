import numpy as np

def compute_precision_at_1_single(gt_files, predicted_files):
    """
    Computes Precision@1 for a single bug.
    
    Parameters:
        gt_files (list or set): Ground-truth files for the bug.
        predicted_files (list): Predicted files, ranked by relevance.
    
    Returns:
        float: Precision@1 value (1.0, 0.0, or NaN).
    """
    if len(gt_files) == 1:
        return 1.0 if predicted_files and predicted_files[0] in gt_files else 0.0
    else:
        return np.nan  # Not applicable

def compute_precision_at_2_single(gt_files, predicted_files):
    """
    Computes Precision@2 for a single bug.
    
    Parameters:
        gt_files (list or set): Ground-truth files for the bug.
        predicted_files (list): Predicted files, ranked by relevance.
    
    Returns:
        float: Precision@2 value between 0.0 and 1.0, or NaN.
    """
    if len(gt_files) >= 2:
        top_2_preds = predicted_files[:2]
        correct = sum(1 for pred in top_2_preds if pred in gt_files)
        return correct / 2.0
    else:
        return np.nan  # Not applicable
    
def compute_precision_at_k(gt_files, predicted_files, k=2):
    """
    Computes Precision@k for a single bug.
    
    Parameters:
        gt_files (list or set): Ground-truth files for the bug.
        predicted_files (list): Predicted files, ranked by relevance.
    
    Returns:
        float: Precision@k value between 0.0 and 1.0, or NaN.
    """
    if not gt_files:
        return np.nan
    
    top_k_preds = predicted_files[:k]
    correct = len(set(gt_files).intersection(top_k_preds))
    return correct / k


def compute_recall_at_1_single(gt_files, predicted_files):
    """
    Computes Recall@1 for a single bug.
    
    Parameters:
        gt_files (list or set): Ground-truth files for the bug.
        predicted_files (list): Predicted files, ranked by relevance.
    
    Returns:
        float: Recall@1 value (1.0, 0.0, or NaN).
    """
    if len(gt_files) == 1:
        return 1.0 if predicted_files and predicted_files[0] in gt_files else 0.0
    else:
        return np.nan  # Not applicable

def compute_recall_at_2_single(gt_files, predicted_files):
    """
        Computes Recall@2 for a single bug.
        
        Parameters:
            gt_files (list or set): Ground-truth files for the bug.
            predicted_files (list): Predicted files, ranked by relevance.
        
        Returns:
            float: Recall@2 value between 0.0 and 1.0, or NaN.
        """
    if len(gt_files) >= 2:
        top_2_preds = predicted_files[:2]
        # Number of unique ground-truth files in top 2 predictions
        correct = len(set(top_2_preds).intersection(set(gt_files)))
        return correct / len(gt_files)
    else:
        return np.nan  # Not applicable
    
def compute_recall_at_k(gt_files, predicted_files, k=1):
    """
        Computes Recall@k for a single bug.
        
        Parameters:
            gt_files (list or set): Ground-truth files for the bug.
            predicted_files (list): Predicted files, ranked by relevance.
        
        Returns:
            float: Recall@k value between 0.0 and 1.0, or NaN.
        """
    if not gt_files:
        return np.nan
    
    top_k_preds = predicted_files[:k]
    # Number of unique ground-truth files in top k predictions
    correct = len(set(gt_files).intersection(set(top_k_preds)))
    return correct / len(gt_files)

def compute_average_precision(gt_files, predicted_files):
    """
    Computes Average Precision (AP) for a single bug report.

    Parameters:
        gt_files (list or set): Ground-truth files for the bug.
        predicted_files (list): Predicted files, ranked by relevance.

    Returns:
        float: Average Precision (AP) value, or np.nan if no ground-truth files.
    """
    if not gt_files:
        return np.nan  # Undefined if no ground-truth files

    gt_files_set = set(gt_files)
    num_relevant = len(gt_files_set)
    if num_relevant == 0:
        return np.nan  # Avoid division by zero

    hits = 0
    sum_precisions = 0.0

    for idx, pred in enumerate(predicted_files, start=1):
        if pred in gt_files_set:
            hits += 1
            precision_at_k = hits / idx
            sum_precisions += precision_at_k

    if hits == 0:
        return 0.0

    average_precision = sum_precisions / num_relevant
    return average_precision

def compute_f1_score(precision, recall):
    """
    Computes the F1 Score given Precision and Recall.
    
    Parameters:
        precision (float): Precision value.
        recall (float): Recall value.
    
    Returns:
        float: F1 Score or np.nan if undefined.
    """
    if np.isnan(precision) or np.isnan(recall):
        return np.nan
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_f1_at_k(precisionatk, recallatk, k=5):
    """
    Computes the F1 score at a given k value.
    The F1 score is the harmonic mean of precision and recall, providing a single
    metric that balances both precision and recall.
    Args:
        precisionatk (float): The precision at k value.
        recallatk (float): The recall at k value.
        k (int, optional): The k value for which the F1 score is computed. Default is 5.
    Returns:
        float: The F1 score at the given k value.
    Raises:
        ValueError: If precisionatk or recallatk is zero, as this would result in division by zero.
    """
    if (precisionatk == 0 or recallatk == 0):
        return 0
    return 2 * ((precisionatk * recallatk) / (precisionatk + recallatk))

def hit_rate_at_k(gt_files, predicted_files, k=5):
    """
    Calculates if at least one ground truth file appears in the top k

        Parameters:
            gt_files (list or set): Ground-truth files for the bug.
            predicted_files (list): Predicted files, ranked by relevance.
            k (int): Number of top predictions to consider.
        
        Returns:
            float: Hit rate at k (1.0 if at least one ground-truth file is in the top k predictions, otherwise 0.0).
    """
    topk = predicted_files[:k]
    return 1.0 if len(set(gt_files).intersection(topk)) > 0 else 0.0