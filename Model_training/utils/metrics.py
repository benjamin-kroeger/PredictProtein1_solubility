import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score, balanced_accuracy_score
def compute_metrics(y_true, y_pred_prob) -> dict:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.cpu().numpy()

    y_pred = (y_pred_prob > 0.5).astype(int)
    metric_dict = {
        'acc': accuracy_score(y_true=y_true, y_pred=y_pred),
        'bal_acc': balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
        'f1': f1_score(y_true=y_true, y_pred=y_pred),
        'recall': recall_score(y_true=y_true, y_pred=y_pred),
        'precision': precision_score(y_true=y_true, y_pred=y_pred),
        'auc': roc_auc_score(y_true=y_true, y_score=y_pred_prob),
        'mcc': matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    }

    return metric_dict
