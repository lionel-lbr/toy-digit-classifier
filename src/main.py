from pathlib import Path
from typing import List
import random
import math


DIGIT_FILENAMES = [
    "digit_0.txt",
    "digit_1.txt",
    "digit_2.txt",
    "digit_3.txt",
    "digit_4.txt",
    "digit_5.txt",
    "digit_6.txt",
    "digit_7.txt",
    "digit_8.txt",
    "digit_9.txt",
]

DIGITS: List[List[int]] = []

HOT_VECTORS: List[List[int]] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]


def load_digit(filename: str) -> List[int]:
    """
    Load a digit text file from the project's data directory.

    Each file is expected to be a matrix of 0s and 1s with one row per line.
    Returns a flat list of integers (0 or 1).
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Digit file not found: {file_path}")

    digit: List[int] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if any(ch not in {"0", "1"} for ch in stripped):
                raise ValueError(f"Invalid character in digit file: {file_path}")

            digit.extend([int(ch) for ch in stripped])

    if not digit:
        raise ValueError(f"Digit file is empty: {file_path}")

    return digit


def load_all_digits() -> None:
    for f in DIGIT_FILENAMES:
        DIGITS.append(load_digit(f))


def init_weights(n_neurons: int, n_inputs: int):
    """Return (weights, biases) with Xavier/Glorot init."""
    std = math.sqrt(1 / n_inputs)
    W = []
    for _ in range(n_neurons):
        row = [random.gauss(0, std) for _ in range(n_inputs)]
        W.append(row)
    b = [0.0 for _ in range(n_neurons)]
    return W, b


def forward_pass(x: List[int], W: List[List[float]], b: List[float]) -> List[float]:
    """Compute one-layer outputs: weighted sum of inputs plus bias per neuron."""
    if not W:
        return []

    n_inputs = len(W[0])
    if len(x) != n_inputs:
        raise ValueError(f"Expected input length {n_inputs}, got {len(x)}")

    outputs: List[float] = []
    for weights, bias in zip(W, b):
        if len(weights) != n_inputs:
            raise ValueError("Inconsistent weight vector length")

        activation = sum(w * xi for w, xi in zip(weights, x)) + bias
        outputs.append(activation)

    return outputs


def softmax(logits: List[float]) -> List[float]:
    """Convert raw scores to probabilities in a numerically stable way."""
    if not logits:
        return []

    shift = max(logits)
    exps = [math.exp(x - shift) for x in logits]
    denom = sum(exps)
    return [val / denom for val in exps]


def predict_proba(x: List[int], W: List[List[float]], b: List[float]) -> List[float]:
    """Return class probabilities for a single input."""
    logits = forward_pass(x, W, b)
    return softmax(logits)


def predict_digit(x: List[int], W: List[List[float]], b: List[float]) -> int:
    """Return the predicted class index for a single input."""
    probs = predict_proba(x, W, b)
    if not probs:
        raise ValueError("No probabilities computed for prediction")
    return max(range(len(probs)), key=probs.__getitem__)


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

    # Update weights and biases
    for i, (d, weights) in enumerate(zip(delta, W)):
        for j, xj in enumerate(x):
            weights[j] -= lr * d * xj
        b[i] -= lr * d

    return loss


def main() -> None:
    """Simple entry point for the digit classifier project."""
    print("DigitClassifier main executed.")
    load_all_digits()
    W, b = init_weights(10, 64)
    learning_rate = 0.1
    best_train_loss = float("inf")
    patience = 5
    min_delta = 1e-4
    wait = 0
    max_epochs = 1000

    train_indices = list(range(len(DIGITS)))

    for epoch in range(1, max_epochs + 1):
        random.shuffle(train_indices)

        # run one epoch on the training split
        train_losses = []
        for digit_index in train_indices:
            digit = DIGITS[digit_index]
            hot_vector = HOT_VECTORS[digit_index]
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
    for digit_index, digit in enumerate(DIGITS):
        prediction = predict_digit(digit, W, b)
        if prediction == digit_index:
            correct += 1
    accuracy = correct / len(DIGITS)
    print(f"Training set accuracy: {accuracy:.2f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
