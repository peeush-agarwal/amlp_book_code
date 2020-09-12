"""
This module defines functions which are (or used for) Evaluation metrics.
"""

def true_positive(y_true, y_pred):
    """
    Computes count of True Positive
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: count of true positive
    """
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    """
    Computes count of True Negative
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: count of true negative
    """
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    """
    Computes count of False Positive
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: count of false positive
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    """
    Computes count of False Negative
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: count of false negative
    """
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def precision(y_true, y_pred):
    """
    Computes the precision score
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: Precision score for the given values
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return 1.0*tp / (tp + fp)

def recall(y_true, y_pred):
    """
    Computes the recall score
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: Recall score for the given values
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return 1.0*tp / (tp + fn)

def f1_score(y_true, y_pred):
    """
    Computes the F1 score
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: F1 score for the given values
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2*p*r)/(p+r)

def tpr(y_true, y_pred):
    """
    Computes the True Positive Rate (TPR)
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: TPR for the given values
    """
    return recall(y_true, y_pred)

def fpr(y_true, y_pred):
    """
    Computes the False Positive Rate (FPR)
    
    :param y_true: Actual target values
    :param y_pred: Predicted values from the model
    :returns: FPR for the given values
    """
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return 1.0 * fp / (tn + fp)

def log_loss(y_true, y_probs):
    """
    Compute the log loss for the given values
    
    :param y_true: Actual target values
    :param y_probs: Predicted probabilities from the model
    :returns Log loss over all given values
    """
    import numpy as np
    eps = 1e-15
    losses = []
    for yt, yp in zip(y_true, y_probs):
        # Adjust the probability
        # 0 is converted to eps or 1e-15
        # 1 is converted to 1-eps or 1-1e-15
        y_pred = np.clip(yp, eps, 1-eps)
        loss = -1.0 * (yt * np.log(y_pred) + (1-yt)*np.log(1-y_pred))
        losses.append(loss)
    # Mean of all losses
    return np.mean(losses)
