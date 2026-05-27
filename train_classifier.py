#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
from statistics import mean
from importlib import import_module


def load_dataset(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Dataset file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: csv has no header")

        target_key = None
        for key in reader.fieldnames:
            if key.lower() in {"target", "y", "label", "class", "cls"}:
                target_key = key
                break

        if target_key is None:
            raise RuntimeError(f"{path}: target column not found")

        feature_keys = [k for k in reader.fieldnames if k != target_key]
        if len(feature_keys) == 0:
            raise RuntimeError(f"{path}: expected at least 1 feature column")

        x, y = [], []
        for row in reader:
            sample = [float(row[k]) for k in feature_keys]
            x.append(sample)
            y.append(float(int(float(row[target_key]))))

    return x, y


def train_test_split(x, y, test_ratio, seed):
    indexes = list(range(len(x)))
    random.Random(seed).shuffle(indexes)

    split = int(len(x) * (1.0 - test_ratio))
    train_idx = indexes[:split]
    test_idx = indexes[split:]

    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return (x_train, y_train, x_test, y_test)


def fit_scaler(x_train):
    if not x_train:
        raise RuntimeError("dataset is empty, cannot fit scaler")

    dims = len(x_train[0])
    for i, row in enumerate(x_train):
        if len(row) != dims:
            raise RuntimeError(f"inconsistent feature dimensions in training set at row {i}: expected {dims}, got {len(row)}")

    mus = []
    stds = []
    for feature_idx in range(dims):
        values = [row[feature_idx] for row in x_train]
        m = mean(values)
        s = max(1e-8, (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5)
        mus.append(m)
        stds.append(s)

    return mus, stds


def scale_dataset(x, mu, std):
    dims = len(mu)
    for i, row in enumerate(x):
        if len(row) != dims:
            raise RuntimeError(f"feature dimension mismatch at row {i}: expected {dims}, got {len(row)}")

    return [[(row[idx] - mu[idx]) / std[idx] for idx in range(dims)] for row in x]


def precision_recall_f1(y_true, y_pred):
    tp = fp = fn = 0
    for gt, pred in zip(y_true, y_pred):
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_f1(model, x, y):
    y_pred = [1 if model.predict(sample) == 1 else 0 for sample in x]
    return precision_recall_f1(y, y_pred)


def train_epoch(model, x_train, y_train, batch_size, lr):
    idx = list(range(len(x_train)))
    random.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        batch_x = [x_train[i] for i in batch_idx]
        batch_y = [y_train[i] for i in batch_idx]
        model.fit(batch_x, batch_y, lr)


def load_lib():
    root = os.path.dirname(os.path.abspath(__file__))
    build_bin = os.path.join(root, "build", "bin")
    if os.path.isdir(build_bin) and build_bin not in sys.path:
        sys.path.insert(0, build_bin)
    try:
        return import_module("comand7")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Не найден Python-модуль comand7. Сначала соберите C++ библиотеку: make"
        ) from exc


def ensure_model_directory(model_path):
    model_dir = os.path.dirname(model_path)
    if not model_dir:
        return

    if os.path.exists(model_dir) and not os.path.isdir(model_dir):
        backup = f"{model_dir}.old"
        while os.path.exists(backup):
            backup = f"{backup}.old"
        os.rename(model_dir, backup)
        print(f"Предупреждение: {model_dir} был файлом, переименован в {backup} и создан как папка.")

    os.makedirs(model_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train classifier from C++ PyModel")
    parser.add_argument("--d1", default="dataset1.csv", help="path to first dataset")
    parser.add_argument("--d2", default="dataset1.csv", help="path to second dataset")
    parser.add_argument("--split", type=float, default=0.2, help="test ratio")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="model.bin", help="path to save model")
    parser.add_argument("--load", default=None, help="load existing model and continue")
    parser.add_argument("--finetune", default=None, help="optional third dataset for fine-tuning")
    parser.add_argument("--finetune-epochs", type=int, default=50, help="epochs for fine-tuning")
    parser.add_argument("--eval-every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)

    x1, y1 = load_dataset(args.d1)
    x2, y2 = load_dataset(args.d2)

    if not x1:
        raise RuntimeError(f"d1 dataset is empty: {args.d1}")
    if not x2:
        raise RuntimeError(f"d2 dataset is empty: {args.d2}")

    d1_features = len(x1[0])
    if any(len(row) != d1_features for row in x1):
        raise RuntimeError(f"inconsistent feature dimensions in {args.d1}")
    if any(len(row) != d1_features for row in x2):
        raise RuntimeError(f"inconsistent feature dimensions in {args.d2}")
    if len(x2[0]) != d1_features:
        raise RuntimeError(f"feature dimension mismatch: {args.d1} has {d1_features}, {args.d2} has {len(x2[0])}")

    x1_train, y1_train, x1_val, y1_val = train_test_split(x1, y1, args.split, args.seed)
    _, _, x2_val, y2_val = train_test_split(x2, y2, args.split, args.seed + 1)

    mu, std = fit_scaler(x1_train)
    x1_train = scale_dataset(x1_train, mu, std)
    x1_val = scale_dataset(x1_val, mu, std)
    x2_val = scale_dataset(x2_val, mu, std)

    lib = load_lib()
    Model = lib.Model
    model = Model(d1_features, args.hidden1, args.hidden2)

    if args.load:
        model.load(args.load)

    for epoch in range(args.epochs):
        train_epoch(model, x1_train, y1_train, args.batch_size, args.lr)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            f1_d1 = evaluate_f1(model, x1_val, y1_val)
            print(f"epoch={epoch + 1} f1_d1={f1_d1:.4f}")

    f1_d1 = evaluate_f1(model, x1_val, y1_val)
    f1_d2 = evaluate_f1(model, x2_val, y2_val)
    final_score = 0.5 * (f1_d1 + f1_d2)

    print(f"F1_d1={f1_d1:.4f}")
    print(f"F1_d2={f1_d2:.4f}")
    print(f"Final=(0.5*F1_d1+0.5*F1_d2)={final_score:.4f}")
    print("Accepted" if final_score >= 0.55 else "Not accepted")

    if args.finetune is not None:
        x3, y3 = load_dataset(args.finetune)
        if not x3:
            raise RuntimeError(f"finetune dataset is empty: {args.finetune}")
        if any(len(row) != d1_features for row in x3):
            raise RuntimeError(f"inconsistent feature dimensions in {args.finetune}")

        x3_train, y3_train, x3_val, y3_val = train_test_split(x3, y3, args.split, args.seed + 2)
        x3_train = scale_dataset(x3_train, mu, std)
        x3_val = scale_dataset(x3_val, mu, std)

        before = evaluate_f1(model, x3_val, y3_val)
        print(f"Finetune_before_F1={before:.4f}")

        for epoch in range(args.finetune_epochs):
            train_epoch(model, x3_train, y3_train, args.batch_size, args.lr)

        after = evaluate_f1(model, x3_val, y3_val)
        print(f"Finetune_after_F1={after:.4f}")

    ensure_model_directory(args.model)
    model.save(args.model)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
