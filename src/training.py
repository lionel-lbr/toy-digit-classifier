import math
import random
from typing import List, Tuple

from data import NUM_CLASSES, load_dataset
from layer import forward_pass, init_weights, softmax
from model import predict_digit


def cross_entropy_loss(target: List[int], probs: List[float]) -> float:
    """
    Compute cross-entropy loss for a one-hot target vector and predicted probs.

    Expected: target length equals probs length; target contains a single 1.
    """
    if len(target) != len(probs):
        raise ValueError("Target and probability vectors must have same length")

    epsilon = 1e-12  # protect log
    loss = 0.0
    for t, p in zip(target, probs):
        if t not in (0, 1):
            raise ValueError("Target must be one-hot encoded (0 or 1 values)")
        loss -= t * math.log(max(p, epsilon))
    return loss


def gradient_step(
    x: List[int], target: List[int], W: List[List[float]], b: List[float], lr: float
) -> float:
    """
    Single SGD step: forward, loss, then weight/bias update using softmax + cross-entropy.

    Updates W and b in-place. Returns the scalar loss for logging.
    """
    logits = forward_pass(x, W, b)
    probs = softmax(logits)
    loss = cross_entropy_loss(target, probs)

    # For softmax + cross-entropy, dL/dlogits = probs - target
    delta = [p - t for p, t in zip(probs, target)]

    for i, (d, weights) in enumerate(zip(delta, W)):
        for j, xj in enumerate(x):
            weights[j] -= lr * d * xj
        b[i] -= lr * d

    return loss


def train_classifier(
    max_epochs: int = 1000,
    learning_rate: float = 0.1,
    patience: int = 5,
    min_delta: float = 1e-4,
) -> Tuple[List[List[float]], List[float], float]:
    """Train a single-layer softmax classifier and return weights, biases, accuracy."""
    images, hot_vectors = load_dataset()
    if not images:
        raise ValueError("No digit data available to train on")

    n_inputs = len(images[0])
    W, b = init_weights(NUM_CLASSES, n_inputs)
    best_train_loss = float("inf")
    wait = 0

    train_indices = list(range(len(images)))

    for epoch in range(1, max_epochs + 1):
        random.shuffle(train_indices)

        train_losses = []
        for idx in train_indices:
            digit = images[idx]
            hot_vector = hot_vectors[idx]
            loss = gradient_step(digit, hot_vector, W, b, learning_rate)
            train_losses.append(loss)

        avg_train_loss = sum(train_losses) / len(train_losses)
        improved = avg_train_loss < best_train_loss - min_delta
        if improved:
            best_train_loss = avg_train_loss
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch:4d} | avg_loss={avg_train_loss:.6f} "
            f"| best={best_train_loss:.6f} | wait={wait}"
        )

        if wait >= patience:
            print("Stopping: no improvement on training loss")
            break

    correct = 0
    for digit, hot_vector in zip(images, hot_vectors):
        prediction = predict_digit(digit, W, b)
        label_idx = hot_vector.index(1)
        if prediction == label_idx:
            correct += 1
    accuracy = correct / len(images)
    print(f"Training set accuracy: {accuracy:.2f}")

    return W, b, accuracy
