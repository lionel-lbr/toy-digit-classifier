import argparse
from pathlib import Path
from typing import List

from data import load_digit_file
from model import load_model, predict_digit, save_model
from training import train_classifier


def render_image(img: List[int]) -> None:
    """Render a flat 64-length image into 8 lines of '0'/'#' characters."""
    if len(img) != 64:
        raise ValueError(f"Expected 64-length image, got {len(img)}")
    lines = [" 01234567"]
    index = 0
    for i in range(0, 64, 8):
        row = "".join("#" if px else " " for px in img[i : i + 8])
        lines.append(f"{index}{row}")
        index += 1
    print("\n".join(lines))
    print("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digit classifier CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument(
        "--out", required=True, help="Path to save the trained model JSON"
    )
    train_parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maximum training epochs (default: 1000)",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Epochs to wait for loss improvement before stopping (default: 5)",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict a digit")
    predict_parser.add_argument(
        "--model", required=True, help="Path to a trained model JSON file"
    )
    predict_parser.add_argument(
        "--input", required=True, help="Path to a digit image text file"
    )
    predict_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the input digit before showing the prediction",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        out_path = Path(args.out)
        W, b, accuracy = train_classifier(
            max_epochs=args.max_epochs, patience=args.patience
        )
        save_model(out_path, W, b)
        print(f"Model saved to {out_path} (training accuracy {accuracy:.2f})")
        return

    if args.command == "predict":
        model_path = Path(args.model)
        digit_path = Path(args.input)
        W, b = load_model(model_path)
        digit = load_digit_file(digit_path)
        if args.render:
            render_image(digit)
        prediction = predict_digit(digit, W, b)
        print(f"Predicted digit: {prediction}")
        return

    parser.error("No command supplied")


if __name__ == "__main__":
    main()
