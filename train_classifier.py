#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import random
import re
import subprocess
import sys
from pathlib import Path


SPLIT_RE = re.compile(r"[,;\t ]+")


def import_cpp_module():
    try:
        return importlib.import_module("cppnn")
    except ImportError:
        project_dir = Path(__file__).resolve().parent
        subprocess.run(
            ["make", "python", f"PYTHON={sys.executable}"],
            cwd=project_dir,
            check=True,
        )
        importlib.invalidate_caches()
        return importlib.import_module("cppnn")


def split_tokens(line: str) -> list[str]:
    return [token for token in SPLIT_RE.split(line.strip()) if token]


def is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def load_dataset(path: Path) -> tuple[list[list[float]], list[int], list[str]]:
    rows = [split_tokens(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: dataset is empty")

    has_header = not all(is_float(token) for token in rows[0])
    if has_header:
        header = rows[0]
        data_rows = rows[1:]
        target_idx = header.index("target") if "target" in header else len(header) - 1
        feature_indices = [idx for idx, name in enumerate(header) if name.startswith("feature_")]
        if not feature_indices:
            feature_indices = [idx for idx in range(len(header)) if idx != target_idx]
        feature_names = [header[idx] for idx in feature_indices]
    else:
        data_rows = rows
        target_idx = len(rows[0]) - 1
        feature_indices = list(range(target_idx))
        feature_names = [f"feature_{idx}" for idx in feature_indices]

    samples: list[list[float]] = []
    targets: list[int] = []
    expected_width = len(rows[0])

    for line_no, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != expected_width:
            raise ValueError(f"{path}:{line_no}: expected {expected_width} columns, got {len(row)}")
        sample = [float(row[idx]) for idx in feature_indices]
        target = int(float(row[target_idx]))
        if target not in (0, 1):
            raise ValueError(f"{path}:{line_no}: target must be 0 or 1")
        samples.append(sample)
        targets.append(target)

    if len(set(targets)) < 2:
        raise ValueError(f"{path}: both classes 0 and 1 are required")
    return samples, targets, feature_names


def stratified_split(targets: list[int], test_size: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx, target in enumerate(targets):
        by_class[target].append(idx)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train_count = 1
        else:
            train_count = int(round(len(indices) * (1.0 - test_size)))
            train_count = max(1, min(len(indices) - 1, train_count))
        train_indices.extend(indices[:train_count])
        test_indices.extend(indices[train_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def fit_scaler(samples: list[list[float]]) -> tuple[list[float], list[float]]:
    feature_count = len(samples[0])
    means = [0.0] * feature_count
    for sample in samples:
        for idx, value in enumerate(sample):
            means[idx] += value
    means = [value / len(samples) for value in means]

    stds = [0.0] * feature_count
    for sample in samples:
        for idx, value in enumerate(sample):
            diff = value - means[idx]
            stds[idx] += diff * diff
    stds = [math.sqrt(value / len(samples)) or 1.0 for value in stds]
    stds = [value if value > 1e-12 else 1.0 for value in stds]
    return means, stds


def transform(samples: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(value - means[idx]) / stds[idx] for idx, value in enumerate(sample)] for sample in samples]


def binary_f1(targets: list[int], predictions: list[int]) -> float:
    tp = sum(1 for target, pred in zip(targets, predictions) if target == 1 and pred == 1)
    fp = sum(1 for target, pred in zip(targets, predictions) if target == 0 and pred == 1)
    fn = sum(1 for target, pred in zip(targets, predictions) if target == 1 and pred == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def accuracy(targets: list[int], predictions: list[int]) -> float:
    return sum(1 for target, pred in zip(targets, predictions) if target == pred) / len(targets)


def predict_by_threshold(probabilities: list[float], threshold: float) -> list[int]:
    return [1 if value >= threshold else 0 for value in probabilities]


def best_threshold(targets: list[int], probabilities: list[float]) -> tuple[float, float]:
    best = (0.5, binary_f1(targets, predict_by_threshold(probabilities, 0.5)))
    candidates = [idx / 100.0 for idx in range(1, 100)]
    for threshold in candidates:
        score = binary_f1(targets, predict_by_threshold(probabilities, threshold))
        if score > best[1]:
            best = (threshold, score)
    return best


def default_hidden_layers(input_size: int) -> list[int]:
    return [max(8, input_size * 4), max(4, input_size * 2)]


def train_once(cppnn, x_train, y_train, hidden_layers, args, seed):
    model = cppnn.MlpClassifier(len(x_train[0]), hidden_layers, seed=seed)
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        seed=seed,
        shuffle=True,
        balance_classes=not args.no_balance,
    )
    train_probabilities = model.predict_proba_batch(x_train)
    if args.threshold == "train":
        threshold, train_f1 = best_threshold(y_train, train_probabilities)
    else:
        threshold = 0.5
        train_f1 = binary_f1(y_train, predict_by_threshold(train_probabilities, threshold))
    return model, threshold, train_f1


def evaluate_dataset(cppnn, path: Path, args) -> dict[str, float | int | str]:
    samples, targets, feature_names = load_dataset(path)
    train_indices, test_indices = stratified_split(targets, args.test_size, args.seed)

    x_train_raw = [samples[idx] for idx in train_indices]
    y_train = [targets[idx] for idx in train_indices]
    x_test_raw = [samples[idx] for idx in test_indices]
    y_test = [targets[idx] for idx in test_indices]

    means, stds = fit_scaler(x_train_raw)
    x_train = transform(x_train_raw, means, stds)
    x_test = transform(x_test_raw, means, stds)

    hidden_layers = args.hidden if args.hidden is not None else default_hidden_layers(len(feature_names))
    best_model = None
    threshold = 0.5
    best_train_f1 = -1.0
    for restart in range(args.restarts):
        model, model_threshold, train_f1 = train_once(
            cppnn,
            x_train,
            y_train,
            hidden_layers,
            args,
            args.seed + restart,
        )
        if train_f1 > best_train_f1:
            best_model = model
            threshold = model_threshold
            best_train_f1 = train_f1

    probabilities = best_model.predict_proba_batch(x_test)
    predictions = predict_by_threshold(probabilities, threshold)
    f1 = binary_f1(y_test, predictions)
    acc = accuracy(y_test, predictions)

    return {
        "path": str(path),
        "rows": len(samples),
        "features": len(feature_names),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "threshold": threshold,
        "train_f1": best_train_f1,
        "test_f1": f1,
        "test_accuracy": acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate a C++ neural-network binary classifier on tabular datasets.",
    )
    parser.add_argument("datasets", nargs="+", type=Path, help="Dataset files with feature_* columns and target column")
    parser.add_argument("--epochs", type=int, default=1500, help="Training epochs per restart")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--hidden", type=int, nargs="*", default=None, help="Hidden layer sizes, for example: --hidden 16 8")
    parser.add_argument("--restarts", type=int, default=3, help="Train several initializations and keep the best by train F1")
    parser.add_argument("--threshold", choices=("train", "fixed"), default="train", help="Tune threshold on train split or use 0.5")
    parser.add_argument("--no-balance", action="store_true", help="Disable class-balanced loss weights")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.restarts <= 0:
        raise ValueError("restarts must be positive")

    cppnn = import_cpp_module()
    results = [evaluate_dataset(cppnn, path, args) for path in args.datasets]

    for idx, result in enumerate(results, start=1):
        print(
            f"d{idx}: file={result['path']} rows={result['rows']} features={result['features']} "
            f"train/test={result['train_rows']}/{result['test_rows']} "
            f"threshold={result['threshold']:.2f} train_f1={result['train_f1']:.4f} "
            f"test_f1={result['test_f1']:.4f} test_accuracy={result['test_accuracy']:.4f}"
        )

    if len(results) == 2:
        final_score = 0.5 * float(results[0]["test_f1"]) + 0.5 * float(results[1]["test_f1"])
        print(f"final_score=0.5*F1(d1)+0.5*F1(d2)={final_score:.4f}")
    elif len(results) > 1:
        mean_f1 = sum(float(result["test_f1"]) for result in results) / len(results)
        print(f"mean_test_f1={mean_f1:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
