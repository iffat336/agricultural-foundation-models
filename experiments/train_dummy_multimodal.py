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
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing runtime dependency. Install the packages in requirements.txt before running "
        "the dummy multimodal training script."
    ) from exc

from data.synthetic_multimodal import SyntheticDataConfig, build_synthetic_splits
from models.multimodal_baseline import ImageOnlyClassifier, MultimodalFusionClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dummy multimodal baselines on synthetic agricultural data.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="outputs/dummy_training",
        help="Directory for dummy training logs and summaries.",
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_multimodal: bool,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device)
        weather = batch["weather"].to(device)
        geo = batch["geo"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        if use_multimodal:
            logits = model(image, weather, geo)
        else:
            logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * label.size(0)

    return running_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_multimodal: bool,
) -> dict[str, float]:
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            weather = batch["weather"].to(device)
            geo = batch["geo"].to(device)
            labels = batch["label"].to(device)

            if use_multimodal:
                logits = model(image, weather, geo)
            else:
                logits = model(image)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_predictions)
    y_prob = torch.cat(all_probabilities)

    return {
        "score": accuracy_score(y_true.numpy(), y_pred.numpy()),
        "f1": f1_score(y_true.numpy(), y_pred.numpy(), average="macro"),
        "ece": expected_calibration_error(y_prob, y_true),
        "log_loss": log_loss(y_true.numpy(), y_prob.numpy(), labels=list(range(y_prob.shape[1]))),
    }


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    use_multimodal: bool,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_multimodal)
        val_metrics = evaluate_model(model, val_loader, device, use_multimodal)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": val_metrics["score"],
                "val_f1": val_metrics["f1"],
                "val_ece": val_metrics["ece"],
                "val_log_loss": val_metrics["log_loss"],
            }
        )

    return history


def build_result_rows(
    model_name: str,
    note: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    use_multimodal: bool,
) -> list[dict[str, object]]:
    split_names = {
        "val": "Validation",
        "region_shift": "Region transfer",
        "year_shift": "Year transfer",
        "missing_modality": "Missing modality",
    }
    rows = []

    for split_key, split_label in split_names.items():
        metrics = evaluate_model(model, loaders[split_key], device, use_multimodal)
        rows.append(
            {
                "run_name": f"{model_name}_{split_key}",
                "use_case": "Synthetic multimodal crop prototype",
                "split": split_label,
                "model_variant": model_name,
                "score": round(metrics["score"], 4),
                "f1": round(metrics["f1"], 4),
                "ece": round(metrics["ece"], 4),
                "notes": note,
            }
        )

    return rows


def write_summary(history_by_model: dict[str, list[dict[str, float]]], result_df: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Dummy Multimodal Prototype Summary", ""]
    lines.append("This report comes from synthetic data and is intended to demonstrate ML workflow and library usage, not real agricultural performance.")
    lines.append("")

    for model_name, history in history_by_model.items():
        last = history[-1]
        lines.append(f"## {model_name}")
        lines.append(
            f"- Final validation score: {last['val_score']:.3f}"
            f", F1: {last['val_f1']:.3f}"
            f", ECE: {last['val_ece']:.3f}"
            f", train_loss: {last['train_loss']:.3f}"
        )
        lines.append("")

    lines.append("## Benchmark-compatible rows")
    lines.append("")
    lines.append("| run_name | split | model_variant | score | f1 | ece | notes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for _, row in result_df.iterrows():
        lines.append(
            f"| {row['run_name']} | {row['split']} | {row['model_variant']} | "
            f"{row['score']:.4f} | {row['f1']:.4f} | {row['ece']:.4f} | {row['notes']} |"
        )
    lines.append("")

    (output_dir / "dummy_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        num_classes=cfg.num_classes,
    ).to(device)

    history_by_model = {
        "dummy_image_baseline": fit_model(
            image_model,
            loaders["train"],
            loaders["val"],
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_multimodal=False,
        ),
        "dummy_multimodal_fusion": fit_model(
            multimodal_model,
            loaders["train"],
            loaders["val"],
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_multimodal=True,
        ),
    }

    results = []
    results.extend(
        build_result_rows(
            model_name="dummy_image_baseline",
            note="Synthetic-data image-only prototype for workflow demonstration",
            model=image_model,
            loaders=loaders,
            device=device,
            use_multimodal=False,
        )
    )
    results.extend(
        build_result_rows(
            model_name="dummy_multimodal_fusion",
            note="Synthetic-data multimodal fusion prototype for workflow demonstration",
            model=multimodal_model,
            loaders=loaders,
            device=device,
            use_multimodal=True,
        )
    )

    result_df = pd.DataFrame(results)
    history_frames = []
    for model_name, history in history_by_model.items():
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, "model_variant", model_name)
        history_frames.append(history_frame)

    pd.concat(history_frames, ignore_index=True).to_csv(output_dir / "dummy_training_history.csv", index=False)
    result_df.to_csv(output_dir / "dummy_training_runs.csv", index=False)
    write_summary(history_by_model, result_df, output_dir)

    print(f"Saved dummy training results to {output_dir}")
    print(f"- {output_dir / 'dummy_training_history.csv'}")
    print(f"- {output_dir / 'dummy_training_runs.csv'}")
    print(f"- {output_dir / 'dummy_training_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
