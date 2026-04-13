from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing runtime dependency. Install the packages in requirements.txt before running "
        "the synthetic multimodal training script."
    ) from exc

from data.synthetic_multimodal import SyntheticDataConfig, build_synthetic_splits
from models.multimodal_baseline import (
    ImageOnlyClassifier,
    MultimodalFusionClassifier,
    PhysicsAwareMultimodalClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train domain-inspired synthetic agricultural baselines with benchmark-compatible outputs."
    )
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=0.25,
        help="Weight for the auxiliary physics-aware regression loss.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/synthetic_study",
        help="Directory for synthetic training logs and summaries.",
    )
    parser.add_argument(
        "--write-log",
        default="data/experiment_runs.csv",
        help="Benchmark-compatible CSV log that the app can load by default.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probabilities.device)
    ece = torch.zeros(1, device=probabilities.device)

    for idx in range(num_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        proportion = in_bin.float().mean()
        if proportion.item() > 0:
            accuracy = accuracies[in_bin].float().mean()
            avg_confidence = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence - accuracy) * proportion

    return float(ece.item())


def forward_model(model: nn.Module, batch: dict[str, torch.Tensor], kind: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    image = batch["image"]
    if kind == "image":
        return model(image), None
    if kind == "physics":
        logits, physics_pred = model(
            image,
            batch["weather"],
            batch["geo"],
            batch["management"],
            batch["text"],
        )
        return logits, physics_pred
    logits = model(
        image,
        batch["weather"],
        batch["geo"],
        batch["management"],
        batch["text"],
    )
    return logits, None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    kind: str,
    physics_weight: float,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_physics = 0.0

    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        optimizer.zero_grad()
        logits, physics_pred = forward_model(model, batch, kind)
        classification_loss = criterion(logits, batch["label"])
        loss = classification_loss
        physics_loss_value = 0.0
        if physics_pred is not None:
            physics_loss = F.mse_loss(physics_pred, batch["physics_target"])
            physics_loss_value = float(physics_loss.item())
            loss = loss + physics_weight * physics_loss
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * batch["label"].size(0)
        running_physics += physics_loss_value * batch["label"].size(0)

    return {
        "train_loss": running_loss / len(loader.dataset),
        "train_physics_loss": running_physics / len(loader.dataset),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str,
) -> dict[str, float]:
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_physics_true = []
    all_physics_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            logits, physics_pred = forward_model(model, batch, kind)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(batch["label"].cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())
            if physics_pred is not None:
                all_physics_true.append(batch["physics_target"].cpu())
                all_physics_pred.append(physics_pred.cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_predictions)
    y_prob = torch.cat(all_probabilities)

    metrics = {
        "score": accuracy_score(y_true.numpy(), y_pred.numpy()),
        "f1": f1_score(y_true.numpy(), y_pred.numpy(), average="macro"),
        "ece": expected_calibration_error(y_prob, y_true),
        "log_loss": log_loss(y_true.numpy(), y_prob.numpy(), labels=list(range(y_prob.shape[1]))),
    }

    if all_physics_pred:
        metrics["physics_mae"] = mean_absolute_error(
            torch.cat(all_physics_true).numpy().ravel(),
            torch.cat(all_physics_pred).numpy().ravel(),
        )
    else:
        metrics["physics_mae"] = float("nan")
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    kind: str,
    physics_weight: float,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, kind, physics_weight)
        val_metrics = evaluate_model(model, val_loader, device, kind)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_physics_loss": train_metrics["train_physics_loss"],
                "val_score": val_metrics["score"],
                "val_f1": val_metrics["f1"],
                "val_ece": val_metrics["ece"],
                "val_log_loss": val_metrics["log_loss"],
                "val_physics_mae": val_metrics["physics_mae"],
            }
        )

    return history


def build_result_rows(
    model_name: str,
    note: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    kind: str,
) -> list[dict[str, object]]:
    split_names = {
        "region_shift": "Region transfer",
        "year_shift": "Year transfer",
        "climate_stress": "Climate-stress transfer",
        "sensor_shift": "Sensor transfer",
        "missing_modality": "Missing modality",
    }
    rows = []

    for split_key, split_label in split_names.items():
        metrics = evaluate_model(model, loaders[split_key], device, kind)
        extra = ""
        if metrics["physics_mae"] == metrics["physics_mae"]:
            extra = f"; physics_mae={metrics['physics_mae']:.3f}"
        rows.append(
            {
                "run_name": f"{model_name}_{split_key}",
                "use_case": "Domain-inspired synthetic agricultural benchmark",
                "split": split_label,
                "model_variant": model_name,
                "score": round(metrics["score"], 4),
                "f1": round(metrics["f1"], 4),
                "ece": round(metrics["ece"], 4),
                "notes": f"{note}{extra}",
            }
        )

    return rows


def write_summary(history_by_model: dict[str, list[dict[str, float]]], result_df: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Domain-Inspired Synthetic Agricultural Study", ""]
    lines.append(
        "This report is based on reproducible synthetic data with domain-inspired crop, weather, region, management, and shift structure. "
        "It is meant to demonstrate a realistic early-stage ML workflow, not to claim real agricultural performance."
    )
    lines.append("")
    lines.append("## Why This Is Stronger Than A Placeholder Demo")
    lines.append("")
    lines.append("- It trains three actual models instead of hand-writing benchmark numbers.")
    lines.append("- It evaluates multiple OOD conditions: region, year, climate stress, sensor shift, and missing modality.")
    lines.append("- It includes a physics-aware model with an auxiliary scientific-consistency target.")
    lines.append("- It produces benchmark-compatible outputs that the app can load directly.")
    lines.append("")

    for model_name, history in history_by_model.items():
        last = history[-1]
        lines.append(f"## {model_name}")
        summary_line = (
            f"- Final validation score: {last['val_score']:.3f}, "
            f"F1: {last['val_f1']:.3f}, "
            f"ECE: {last['val_ece']:.3f}, "
            f"log_loss: {last['val_log_loss']:.3f}, "
            f"train_loss: {last['train_loss']:.3f}"
        )
        if last["val_physics_mae"] == last["val_physics_mae"]:
            summary_line += f", physics_mae: {last['val_physics_mae']:.3f}"
        lines.append(summary_line)
        lines.append("")

    lines.append("## Benchmark-Compatible Rows")
    lines.append("")
    lines.append("| run_name | split | model_variant | score | f1 | ece | notes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for _, row in result_df.iterrows():
        lines.append(
            f"| {row['run_name']} | {row['split']} | {row['model_variant']} | "
            f"{row['score']:.4f} | {row['f1']:.4f} | {row['ece']:.4f} | {row['notes']} |"
        )
    lines.append("")

    (output_dir / "synthetic_study_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_log_path = Path(args.write_log)
    write_log_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SyntheticDataConfig()
    datasets = build_synthetic_splits(seed=args.seed)
    loaders = {
        split_name: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split_name == "train"))
        for split_name, dataset in datasets.items()
    }

    image_model = ImageOnlyClassifier(in_channels=cfg.image_channels, num_classes=cfg.num_classes).to(device)
    multimodal_model = MultimodalFusionClassifier(
        in_channels=cfg.image_channels,
        weather_steps=cfg.weather_steps,
        weather_features=cfg.weather_features,
        geo_features=cfg.geo_features,
        management_features=cfg.management_features,
        text_features=cfg.text_features,
        num_classes=cfg.num_classes,
    ).to(device)
    physics_model = PhysicsAwareMultimodalClassifier(
        in_channels=cfg.image_channels,
        weather_steps=cfg.weather_steps,
        weather_features=cfg.weather_features,
        geo_features=cfg.geo_features,
        management_features=cfg.management_features,
        text_features=cfg.text_features,
        num_classes=cfg.num_classes,
    ).to(device)

    history_by_model = {
        "baseline": fit_model(
            image_model,
            loaders["train"],
            loaders["val"],
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            kind="image",
            physics_weight=args.physics_weight,
        ),
        "multimodal_ssl": fit_model(
            multimodal_model,
            loaders["train"],
            loaders["val"],
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            kind="multimodal",
            physics_weight=args.physics_weight,
        ),
        "physics_aware_fm": fit_model(
            physics_model,
            loaders["train"],
            loaders["val"],
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            kind="physics",
            physics_weight=args.physics_weight,
        ),
    }

    results = []
    results.extend(
        build_result_rows(
            model_name="baseline",
            note="Image-only baseline trained on domain-inspired synthetic crop observations",
            model=image_model,
            loaders=loaders,
            device=device,
            kind="image",
        )
    )
    results.extend(
        build_result_rows(
            model_name="multimodal_ssl",
            note="Multimodal fusion model trained on synthetic image, weather, geo, management, and text features",
            model=multimodal_model,
            loaders=loaders,
            device=device,
            kind="multimodal",
        )
    )
    results.extend(
        build_result_rows(
            model_name="physics_aware_fm",
            note="Physics-aware multimodal model with auxiliary scientific-consistency target on synthetic agronomic signals",
            model=physics_model,
            loaders=loaders,
            device=device,
            kind="physics",
        )
    )

    result_df = pd.DataFrame(results)
    history_frames = []
    for model_name, history in history_by_model.items():
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, "model_variant", model_name)
        history_frames.append(history_frame)

    pd.concat(history_frames, ignore_index=True).to_csv(output_dir / "synthetic_study_history.csv", index=False)
    result_df.to_csv(output_dir / "synthetic_study_runs.csv", index=False)
    result_df.to_csv(write_log_path, index=False)
    write_summary(history_by_model, result_df, output_dir)

    print(f"Saved synthetic study results to {output_dir}")
    print(f"- {output_dir / 'synthetic_study_history.csv'}")
    print(f"- {output_dir / 'synthetic_study_runs.csv'}")
    print(f"- {output_dir / 'synthetic_study_summary.md'}")
    print(f"- {write_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
