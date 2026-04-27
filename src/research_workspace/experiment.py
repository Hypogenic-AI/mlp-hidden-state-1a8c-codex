"""End-to-end experiment runner for MLP hidden-state similarity."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import DatasetDict, load_from_disk
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from .similarity import all_similarity_metrics


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"


@dataclass
class DatasetConfig:
    name: str
    path: str
    image_key: str
    input_dim: int
    epochs: int
    train_limit: int
    test_limit: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    """Four-layer MLP with equal-width hidden states for easier comparison."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 4, num_classes: int = 10):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for _ in range(num_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                )
            )
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: Tensor, return_hidden: bool = False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        hidden_states: List[Tensor] = []
        for layer in self.hidden_layers:
            x = layer(x)
            hidden_states.append(x)
        logits = self.output_layer(x)
        if return_hidden:
            return logits, hidden_states
        return logits


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(0)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lower = np.quantile(samples, alpha / 2)
    upper = np.quantile(samples, 1 - alpha / 2)
    return float(lower), float(upper)


def cohens_d_paired(x: Iterable[float], y: Iterable[float]) -> float:
    diffs = np.asarray(list(x), dtype=np.float64) - np.asarray(list(y), dtype=np.float64)
    std = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    if std == 0:
        return 0.0
    return float(diffs.mean() / std)


def permutation_paired_pvalue(x: Iterable[float], y: Iterable[float], n_perm: int = 5000) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    diffs = x_arr - y_arr
    observed = abs(diffs.mean())
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_info() -> List[Dict[str, str | int]]:
    info: List[Dict[str, str | int]] = []
    if not torch.cuda.is_available():
        return info
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info.append(
            {
                "index": idx,
                "name": props.name,
                "memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _stratified_subset_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    chosen, _ = train_test_split(
        np.arange(len(labels)),
        train_size=limit,
        random_state=seed,
        stratify=labels,
    )
    return np.sort(chosen)


def load_full_dataset_arrays(config: DatasetConfig, subset_seed: int) -> Dict[str, np.ndarray]:
    dataset: DatasetDict = load_from_disk(str(ROOT / config.path))
    train_labels_all = np.asarray(dataset["train"]["label"], dtype=np.int64)
    test_labels_all = np.asarray(dataset["test"]["label"], dtype=np.int64)

    train_indices = _stratified_subset_indices(train_labels_all, config.train_limit, subset_seed)
    test_indices = _stratified_subset_indices(test_labels_all, config.test_limit, subset_seed)

    train_split = dataset["train"].select(train_indices.tolist()).with_format("numpy")
    test_split = dataset["test"].select(test_indices.tolist()).with_format("numpy")

    x_train_full = np.asarray(train_split[:][config.image_key], dtype=np.float32)
    x_test = np.asarray(test_split[:][config.image_key], dtype=np.float32)
    if x_train_full.ndim == 3:
        x_train_full = x_train_full[..., None]
    if x_test.ndim == 3:
        x_test = x_test[..., None]
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train_full = np.asarray(train_split["label"], dtype=np.int64)
    y_test = np.asarray(test_split["label"], dtype=np.int64)

    return {
        "x_train_full": x_train_full,
        "y_train_full": y_train_full,
        "x_test": x_test,
        "y_test": y_test,
    }


def split_dataset_arrays(full_arrays: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    x_train_full = full_arrays["x_train_full"]
    y_train_full = full_arrays["y_train_full"]
    x_test = full_arrays["x_test"]
    y_test = full_arrays["y_test"]

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=0.15,
        random_state=seed,
        stratify=y_train_full,
    )

    arrays = {
        "x_train": x_train_full[train_idx],
        "y_train": y_train_full[train_idx],
        "x_val": x_train_full[val_idx],
        "y_val": y_train_full[val_idx],
        "x_test": x_test,
        "y_test": y_test,
    }
    return arrays


def summarize_dataset(arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    y_train = arrays["y_train"]
    x_train = arrays["x_train"]
    unique, counts = np.unique(y_train, return_counts=True)
    return {
        "train_shape": list(x_train.shape),
        "val_shape": list(arrays["x_val"].shape),
        "test_shape": list(arrays["x_test"].shape),
        "missing_values": int(np.isnan(x_train).sum()),
        "pixel_mean": float(x_train.mean()),
        "pixel_std": float(x_train.std()),
        "label_distribution_train": {int(k): int(v) for k, v in zip(unique, counts)},
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def evaluate(model: MLP, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            losses.append(loss.item())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(batch_y.cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    return float(np.mean(losses)), float(accuracy_score(y_true, y_pred))


def train_single_model(config: DatasetConfig, arrays: Dict[str, np.ndarray], seed: int, hidden_dim: int, batch_size: int) -> Dict[str, object]:
    set_seed(seed)
    device = get_device()
    model = MLP(input_dim=config.input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    train_loader = make_loader(arrays["x_train"], arrays["y_train"], batch_size=batch_size, shuffle=True)
    val_loader = make_loader(arrays["x_val"], arrays["y_val"], batch_size=batch_size, shuffle=False)
    test_loader = make_loader(arrays["x_test"], arrays["y_test"], batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_acc = float("-inf")
    patience = 4
    bad_epochs = 0
    history = []
    start = time.time()

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            train_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
            train_labels.append(batch_y.detach().cpu().numpy())

        y_true_train = np.concatenate(train_labels)
        y_pred_train = np.concatenate(train_preds)
        train_acc = float(accuracy_score(y_true_train, y_pred_train))
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - start
    return {
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_sec": elapsed,
    }


def extract_hidden_states(model: MLP, x: np.ndarray, batch_size: int) -> List[np.ndarray]:
    device = get_device()
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False, num_workers=0)
    collected: List[List[np.ndarray]] = [[] for _ in range(len(model.hidden_layers))]
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            _, hidden = model(batch_x, return_hidden=True)
            for idx, layer_output in enumerate(hidden):
                collected[idx].append(layer_output.cpu().numpy())
    return [np.concatenate(parts, axis=0) for parts in collected]


def compute_within_model_similarity(hidden_states: List[np.ndarray], dataset: str, seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i in range(len(hidden_states)):
        for j in range(len(hidden_states)):
            metrics = all_similarity_metrics(hidden_states[i], hidden_states[j])
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "layer_i": i + 1,
                        "layer_j": j + 1,
                        "layer_distance": abs(i - j),
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return rows


def compute_cross_seed_similarity(hidden_by_seed: Dict[int, List[np.ndarray]], dataset: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seeds = sorted(hidden_by_seed)
    for idx_a in range(len(seeds)):
        for idx_b in range(idx_a + 1, len(seeds)):
            seed_a = seeds[idx_a]
            seed_b = seeds[idx_b]
            hidden_a = hidden_by_seed[seed_a]
            hidden_b = hidden_by_seed[seed_b]
            for layer_i, acts_a in enumerate(hidden_a):
                for layer_j, acts_b in enumerate(hidden_b):
                    metrics = all_similarity_metrics(acts_a, acts_b)
                    for metric_name, value in metrics.items():
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed_a": seed_a,
                                "seed_b": seed_b,
                                "layer_i": layer_i + 1,
                                "layer_j": layer_j + 1,
                                "same_depth": layer_i == layer_j,
                                "metric": metric_name,
                                "value": value,
                            }
                        )
    return rows


def fit_linear_probes(hidden_train: List[np.ndarray], hidden_test: List[np.ndarray], y_train: np.ndarray, y_test: np.ndarray, dataset: str, seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample_cap = min(12000, len(y_train))
    train_x = hidden_train
    test_x = hidden_test
    train_idx = np.arange(sample_cap)
    test_cap = min(5000, len(y_test))
    test_idx = np.arange(test_cap)
    for layer_idx, (layer_train, layer_test) in enumerate(zip(train_x, test_x), start=1):
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(layer_train[train_idx], y_train[train_idx])
        pred = clf.predict(layer_test[test_idx])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "layer": layer_idx,
                "probe_accuracy": float(accuracy_score(y_test[test_idx], pred)),
            }
        )
    return rows


def plot_similarity_heatmaps(sim_df: pd.DataFrame) -> None:
    for (dataset, seed, metric), frame in sim_df.groupby(["dataset", "seed", "metric"]):
        pivot = frame.pivot(index="layer_i", columns="layer_j", values="value")
        plt.figure(figsize=(5, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0)
        plt.title(f"{dataset} seed={seed} {metric.upper()} within-model similarity")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{dataset}_seed{seed}_{metric}_heatmap.png", dpi=180)
        plt.close()


def plot_distance_trends(sim_df: pd.DataFrame) -> None:
    agg = sim_df.groupby(["dataset", "metric", "layer_distance"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        count=("value", "count"),
    )
    agg["sem"] = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
    for dataset, frame in agg.groupby("dataset"):
        plt.figure(figsize=(6, 4))
        for metric, metric_frame in frame.groupby("metric"):
            plt.plot(metric_frame["layer_distance"], metric_frame["mean"], marker="o", label=metric.upper())
            plt.fill_between(
                metric_frame["layer_distance"],
                metric_frame["mean"] - metric_frame["sem"],
                metric_frame["mean"] + metric_frame["sem"],
                alpha=0.15,
            )
        plt.ylim(0, 1.02)
        plt.xlabel("Layer distance")
        plt.ylabel("Similarity")
        plt.title(f"{dataset}: similarity vs layer distance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{dataset}_distance_trend.png", dpi=180)
        plt.close()


def plot_probe_trends(probe_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=probe_df, x="layer", y="probe_accuracy", hue="dataset", marker="o", errorbar="sd")
    plt.ylim(0, 1.0)
    plt.title("Linear probe accuracy by hidden layer")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probe_accuracy_by_layer.png", dpi=180)
    plt.close()


def make_summary_tables(
    metrics_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    probe_df: pd.DataFrame,
) -> Dict[str, object]:
    summaries: Dict[str, object] = {}

    test_acc_summary = (
        metrics_df.groupby("dataset")["test_acc"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )
    summaries["test_accuracy_summary"] = test_acc_summary

    distance_rows = []
    for (dataset, metric), frame in sim_df.groupby(["dataset", "metric"]):
        adjacent = frame[frame["layer_distance"] == 1]["value"].to_numpy()
        distant = frame[frame["layer_distance"] >= 2]["value"].to_numpy()
        p_value = permutation_paired_pvalue(adjacent[: len(distant)], distant[: len(adjacent)]) if len(adjacent) and len(distant) else float("nan")
        ci_low, ci_high = bootstrap_ci(adjacent)
        distance_rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "adjacent_mean": float(adjacent.mean()),
                "non_adjacent_mean": float(distant.mean()),
                "adjacent_ci_low": ci_low,
                "adjacent_ci_high": ci_high,
                "difference": float(adjacent.mean() - distant.mean()),
                "cohens_d": cohens_d_paired(adjacent[: len(distant)], distant[: len(adjacent)]) if len(adjacent) and len(distant) else float("nan"),
                "p_value": p_value,
            }
        )
    summaries["distance_summary"] = distance_rows

    cross_rows = []
    if not cross_df.empty:
        for (dataset, metric), frame in cross_df.groupby(["dataset", "metric"]):
            same = frame[frame["same_depth"]]["value"].to_numpy()
            diff = frame[~frame["same_depth"]]["value"].to_numpy()
            matched = min(len(same), len(diff))
            p_value = permutation_paired_pvalue(same[:matched], diff[:matched]) if matched else float("nan")
            cross_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "same_depth_mean": float(same.mean()),
                    "off_depth_mean": float(diff.mean()),
                    "difference": float(same.mean() - diff.mean()),
                    "cohens_d": cohens_d_paired(same[:matched], diff[:matched]) if matched else float("nan"),
                    "p_value": p_value,
                }
            )
    summaries["cross_seed_summary"] = cross_rows

    probe_rows = (
        probe_df.groupby(["dataset", "layer"])["probe_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .round(4)
        .to_dict(orient="records")
    )
    summaries["probe_summary"] = probe_rows
    return summaries


def write_environment(batch_size: int, hidden_dim: int, seeds: List[int]) -> None:
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
            text=True,
        ).strip()
    except Exception:
        nvidia_smi = "unavailable"

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "gpu_info": get_gpu_info(),
        "nvidia_smi": nvidia_smi,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "seeds": seeds,
    }
    (RESULTS_DIR / "environment.json").write_text(json.dumps(env, indent=2))


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    write_environment(batch_size=args.batch_size, hidden_dim=args.hidden_dim, seeds=seeds)

    dataset_configs = [
        DatasetConfig("mnist", "datasets/mnist", "image", 28 * 28, epochs=args.epochs_mnist, train_limit=args.train_samples_mnist, test_limit=args.test_samples_mnist),
        DatasetConfig("fashion_mnist", "datasets/fashion_mnist", "image", 28 * 28, epochs=args.epochs_fashion, train_limit=args.train_samples_fashion, test_limit=args.test_samples_fashion),
        DatasetConfig("cifar10", "datasets/cifar10", "img", 32 * 32 * 3, epochs=args.epochs_cifar, train_limit=args.train_samples_cifar, test_limit=args.test_samples_cifar),
    ]

    dataset_summaries = {}
    metrics_rows: List[Dict[str, object]] = []
    sim_rows: List[Dict[str, object]] = []
    cross_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []

    for config in dataset_configs:
        full_arrays = load_full_dataset_arrays(config, subset_seed=seeds[0])
        arrays = split_dataset_arrays(full_arrays, seed=seeds[0])
        dataset_summaries[config.name] = summarize_dataset(arrays)
        hidden_test_by_seed: Dict[int, List[np.ndarray]] = {}

        for seed in seeds:
            seed_arrays = split_dataset_arrays(full_arrays, seed=seed)
            outcome = train_single_model(config, seed_arrays, seed, args.hidden_dim, args.batch_size)
            model: MLP = outcome.pop("model")  # type: ignore[assignment]
            metrics_rows.append(
                {
                    "dataset": config.name,
                    "seed": seed,
                    **{k: v for k, v in outcome.items() if k != "history"},
                }
            )
            history_path = RESULTS_DIR / f"{config.name}_seed{seed}_history.json"
            history_path.write_text(json.dumps(outcome["history"], indent=2))

            test_subset = seed_arrays["x_test"][: args.activation_samples]
            train_subset = seed_arrays["x_train"][: min(args.probe_train_samples, len(seed_arrays["x_train"]))]
            y_train_subset = seed_arrays["y_train"][: min(args.probe_train_samples, len(seed_arrays["y_train"]))]

            hidden_test = extract_hidden_states(model, test_subset, args.batch_size)
            hidden_train = extract_hidden_states(model, train_subset, args.batch_size)
            hidden_test_by_seed[seed] = hidden_test

            sim_rows.extend(compute_within_model_similarity(hidden_test, config.name, seed))
            probe_rows.extend(
                fit_linear_probes(
                    hidden_train,
                    hidden_test,
                    y_train_subset,
                    seed_arrays["y_test"][: args.activation_samples],
                    config.name,
                    seed,
                )
            )

        cross_rows.extend(compute_cross_seed_similarity(hidden_test_by_seed, config.name))

    metrics_df = pd.DataFrame(metrics_rows)
    sim_df = pd.DataFrame(sim_rows)
    cross_df = pd.DataFrame(
        cross_rows,
        columns=["dataset", "seed_a", "seed_b", "layer_i", "layer_j", "same_depth", "metric", "value"],
    )
    probe_df = pd.DataFrame(probe_rows)

    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    sim_df.to_csv(RESULTS_DIR / "within_model_similarity.csv", index=False)
    cross_df.to_csv(RESULTS_DIR / "cross_seed_similarity.csv", index=False)
    probe_df.to_csv(RESULTS_DIR / "probe_accuracy.csv", index=False)
    (RESULTS_DIR / "dataset_summaries.json").write_text(json.dumps(dataset_summaries, indent=2))

    plot_similarity_heatmaps(sim_df)
    plot_distance_trends(sim_df)
    plot_probe_trends(probe_df)

    summary = make_summary_tables(metrics_df, sim_df, cross_df, probe_df)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--activation-samples", type=int, default=1000)
    parser.add_argument("--probe-train-samples", type=int, default=8000)
    parser.add_argument("--epochs-mnist", type=int, default=16)
    parser.add_argument("--epochs-fashion", type=int, default=18)
    parser.add_argument("--epochs-cifar", type=int, default=20)
    parser.add_argument("--train-samples-mnist", type=int, default=20000)
    parser.add_argument("--test-samples-mnist", type=int, default=5000)
    parser.add_argument("--train-samples-fashion", type=int, default=20000)
    parser.add_argument("--test-samples-fashion", type=int, default=5000)
    parser.add_argument("--train-samples-cifar", type=int, default=25000)
    parser.add_argument("--test-samples-cifar", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
