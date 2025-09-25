import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from typing import Union
from torch import Tensor


def metric_regression(
    y_true: Union[npt.NDArray, Tensor],
    y_pred: Union[npt.NDArray, Tensor],
    mean: float = 0,
    std: float = 1,
) -> dict[str, float]:
    """Compute evaluation metrics for regression tasks

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)
        include_mean_std: Whether to include mean and std of predictions

    Returns:
        Dictionary containing:
        - mae: Mean absolute error
        - mse: Mean squared error
        - pearsonr: Pearson correlation coefficient
        - r2: R-squared score
        - mape: Mean absolute percentage error
        - mean_pred (optional): Mean of predictions
        - std_pred (optional): Standard deviation of predictions
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    y_true *= std
    y_true += mean
    y_pred *= std
    y_pred += mean

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "mape": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "r2": float("nan"),
        }

    try:
        if len(y_true) <= 1:
            return {
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "mape": float("nan"),
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "r2": float("nan"),
            }

        epsilon = 1e-10  # Small constant to avoid division by zero
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "pearson_r": pearsonr(y_true, y_pred)[0],
            "pearson_p": pearsonr(y_true, y_pred)[1],
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100,
        }

        return metrics
    except:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "mape": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "r2": float("nan"),
        }


def metric_classification(
    y_true: Union[npt.NDArray, Tensor], y_pred: Union[npt.NDArray, Tensor]
) -> dict[str, float]:
    """Compute evaluation metrics for classification tasks

    Args:
        y_true: Ground truth labels, shape (n_samples,)
        y_pred: Predicted labels or probabilities, shape (n_samples,) or (n_samples, n_classes)

    Returns:
        Dictionary containing:
        - acc: Accuracy score
        - cross_entropy: Cross-entropy loss
        If input contains NaN values, all metrics will be NaN.
    """
    try:
        if isinstance(y_true, Tensor):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.cpu().detach().numpy()

        if len(y_true) <= 1:
            return {
                "acc": float("nan"),
                # "cross_entropy": float("nan"),
            }

        # Check for NaN values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            return {
                "acc": float("nan"),
                # "cross_entropy": float("nan"),
            }

        # Convert y_true to integers (class labels)
        y_true = y_true.astype(np.int32)

        # Handle different y_pred formats
        if y_pred.ndim == 1:
            # If y_pred is 1D, it could be either class labels or probabilities
            if np.issubdtype(y_pred.dtype, np.floating):
                # Floating type - treat as probabilities for binary classification
                y_pred_class = np.round(y_pred).astype(np.int32)
            else:
                # Integer type - treat as class labels
                y_pred_class = y_pred.astype(np.int32)
        else:
            # y_pred is 2D - treat as probability matrix, get class with highest probability
            y_pred_class = np.argmax(y_pred, axis=1)

        # Compute metrics
        try:
            acc = accuracy_score(y_true, y_pred_class)
        except Exception:
            acc = float("nan")

        # # For log_loss, y_pred must be probabilities
        # if y_pred.ndim == 1 and np.issubdtype(y_pred.dtype, np.floating):
        #     # Binary classification probabilities
        #     ce = log_loss(y_true, y_pred)
        # elif y_pred.ndim == 2:
        #     # Multi-class probabilities
        #     ce = log_loss(y_true, y_pred, labels=np.arange(y_pred.shape[-1]))
        # else:
        #     # If y_pred is class labels, we can't compute cross-entropy
        #     ce = float("nan")

        return {
            "acc": acc,
            # "cross_entropy": ce,
        }
    except Exception as e:
        print("error in metric classification", e)
        return {
            "acc": float("nan"),
            # "cross_entropy": float("nan")
        }

def metric_binary_classification(
    y_true: Union[npt.NDArray, Tensor], y_pred: Union[npt.NDArray, Tensor], need_sigmoid: bool = False
) -> dict[str, float]:
    """Compute evaluation metrics for binary classification tasks

    Args:
        y_true: Ground truth labels (0 or 1), shape (n_samples,)
        y_pred: Predicted probabilities or labels, shape (n_samples,) or (n_samples, 2)

    Returns:
        Dictionary containing:
        - acc: Accuracy score
        - cross_entropy: Cross-entropy loss
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - auc: Area Under ROC Curve
        If input contains NaN values, all metrics will be NaN.
    """

    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    if need_sigmoid:
        y_pred = 1 / (1 + np.exp(-1 * y_pred))

    if len(y_true) <= 1:
        return {
            "acc": float("nan"),
            "cross_entropy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }

    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return {
            "acc": float("nan"),
            "cross_entropy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }

    # Convert y_true to binary (0 or 1)
    y_true = y_true.astype(np.int32)
    
    # Handle different y_pred formats
    if y_pred.ndim == 1:
        # Single array: could be probabilities or binary predictions
        if np.issubdtype(y_pred.dtype, np.floating):
            # Probabilities for class 1
            y_prob = y_pred
            y_pred_class = (y_prob >= 0.5).astype(np.int32)
        else:
            # Already binary class predictions
            y_pred_class = y_pred.astype(np.int32)
            y_prob = y_pred_class  # For AUC we'll have issues if we don't have probabilities
    else:
        # 2D array: assumed to be probabilities for both classes
        y_prob = y_pred[:, 1]  # Probability of class 1
        y_pred_class = np.argmax(y_pred, axis=1)

    # Compute basic metrics
    try:
        acc = accuracy_score(y_true, y_pred_class)
        precision = precision_score(y_true, y_pred_class, zero_division=np.nan)
        recall = recall_score(y_true, y_pred_class, zero_division=np.nan)
        f1 = f1_score(y_true, y_pred_class, zero_division=np.nan)
    except Exception as e:
        print("CALC METRIC ERROR", e)
        acc = float("nan")
        precision = float("nan")
        recall = float("nan")
        f1 = float("nan")

    # Compute cross-entropy (log loss)
    try:
        if y_prob.ndim == 1 and np.issubdtype(y_prob.dtype, np.floating):
            ce = log_loss(y_true, y_prob)
        else:
            ce = float("nan")
    except Exception:
        ce = float("nan")

    # Compute AUC
    try:
        if y_prob.ndim == 1 and np.issubdtype(y_prob.dtype, np.floating):
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    return {
        "acc": acc,
        "cross_entropy": ce,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }
