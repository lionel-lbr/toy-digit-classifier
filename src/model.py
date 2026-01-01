import json
from pathlib import Path
from typing import List, Tuple

from layer import forward_pass, softmax


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


def save_model(path: Path, W: List[List[float]], b: List[float]) -> None:
    """Persist model weights and biases to JSON."""
    n_neurons = len(W)
    n_inputs = len(W[0]) if W else 0
    model = {
        "n_neurons": n_neurons,
        "n_inputs": n_inputs,
        "weights": W,
        "bias": b,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(path: Path) -> Tuple[List[List[float]], List[float]]:
    """Load weights and biases from JSON with basic shape validation."""
    data = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {"n_neurons", "n_inputs", "weights", "bias"}
    if not required_keys.issubset(data):
        missing = required_keys - set(data)
        raise ValueError(f"Model file missing keys: {', '.join(sorted(missing))}")

    W = data["weights"]
    b = data["bias"]
    n_neurons = data["n_neurons"]
    n_inputs = data["n_inputs"]

    if len(W) != n_neurons:
        raise ValueError("Weight matrix rows do not match n_neurons in model file")
    if len(b) != n_neurons:
        raise ValueError("Bias vector length does not match n_neurons in model file")
    if W and any(len(row) != n_inputs for row in W):
        raise ValueError("Weight vectors do not match n_inputs in model file")

    return W, b
