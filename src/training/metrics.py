"""
Metrics for evaluating the code correction model
"""
import numpy as np
from transformers import EvalPrediction

def exact_match(predictions, references):
    """
    Calculate exact match between predictions and references.
    
    Args:
        predictions: List of predicted codes
        references: List of reference codes
        
    Returns:
        float: Exact match score
    """
    exact_matches = sum(pred == ref for pred, ref in zip(predictions, references))
    return exact_matches / len(predictions) if predictions else 0

def character_error_rate(predictions, references):
    """
    Calculate character error rate between predictions and references.
    
    Args:
        predictions: List of predicted codes
        references: List of reference codes
        
    Returns:
        float: Character error rate
    """
    total_chars = sum(len(ref) for ref in references)
    edit_distances = []
    
    for pred, ref in zip(predictions, references):
        # Simple Levenshtein distance (character-level edit distance)
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == ref[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_distances.append(dp[m][n])
    
    total_edits = sum(edit_distances)
    return total_edits / total_chars if total_chars else 0

def compute_metrics(eval_preds):
    """
    Compute metrics for the model evaluation.
    
    Args:
        eval_preds: EvalPrediction with predictions and labels
        
    Returns:
        dict: Metrics dictionary
    """
    predictions, labels = eval_preds
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # We need to import the tokenizer here since it's not part of eval_preds
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    
    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute metrics
    em_score = exact_match(decoded_preds, decoded_labels)
    cer = character_error_rate(decoded_preds, decoded_labels)
    
    return {
        "exact_match": em_score,
        "character_error_rate": cer,
    }
