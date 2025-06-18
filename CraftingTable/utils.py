from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    classification_report
)

eval_metrics = {
            "classification": {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1_score": f1_score,
                "roc_auc": roc_auc_score,
                "log_loss": log_loss,
                "classification_report": classification_report
            },
            "regression": {
                "r2_score": r2_score,
                "mean_absolute_error": mean_absolute_error,
                "mean_squared_error": mean_squared_error,
                "root_mean_squared_error": root_mean_squared_error,
                "median_absolute_error": median_absolute_error,
                "explained_variance": explained_variance_score
            }
        }