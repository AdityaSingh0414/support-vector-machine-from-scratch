"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # Convert input to float array
    x = np.asarray(x, dtype=float)

    # Calculate mean for each column
    mean = np.mean(x, axis=0)

    # Calculate standard deviation for each column
    std = np.std(x, axis=0)

    # Subtract mean from every column
    result = x - mean

    # Standardize only columns with non-zero standard deviation
    mask = std != 0

    result[:, mask] = result[:, mask] / std[mask]

    # Constant columns remain centered at zero
    return result

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    w=np.zeros(n_features,dtype=float)
    b=0.0

    return {
        'w':w,
        'b':b
    }

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    w=params['w']
    b=params['b']

    scores= x @ w +b
    return scores

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
     # Positive and zero scores -> +1
    # Negative scores -> -1
    return np.where(scores>=0,1,-1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    return float(max(0.0,1.0-y*score))

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    """Return mean hinge loss + L2 regularization penalty."""
    
    scores = compute_scores(x, params)
    
    hinge_losses = np.maximum(0.0, 1.0 - y * scores)
    
    mean_hinge_loss = np.mean(hinge_losses)
    w=params['w']
    regularization=reg_lambda *np.dot(w,w)
    return float(mean_hinge_loss + regularization)

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    w = params['w']
    b = params['b']

    scores = x @ w + b
    mask = (y * scores) < 1

    dw = -np.sum(
        x[mask] * y[mask, np.newaxis],
        axis=0
    ) / len(y)

    db = -np.sum(y[mask]) / len(y)

    dw += 2 * reg_lambda * w

    return {
        'dw': dw,
        'db': float(db)
    }

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    """Return updated SVM parameters using one gradient-descent step."""

    new_w = params['w'] - learning_rate * grads['dw']
    new_b = params['b'] - learning_rate * grads['db']

    return {
        'w': new_w,
        'b': new_b
    }

# Step 9 - train_svm
import numpy as np

def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    """Fit a linear SVM using full-batch gradient descent."""

    # Step 1: Initialize parameters
    params = initialize_parameters(x.shape[1])

    # Step 2: Training loop
    for _ in range(n_epochs):

        # Compute gradients for the entire dataset
        grads = compute_gradients(
            x,
            y,
            params,
            reg_lambda
        )

        # Update weights and bias
        params = apply_update(
            params,
            grads,
            learning_rate
        )

    return params

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

