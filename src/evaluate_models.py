"""Evaluate two saved ResNet18 checkpoints on the SAME validation split.

Example:
python src/evaluate_models.py --data-dir /kaggle/working \
  --central /kaggle/working/resnet18_centralized_best.pt \
  --federated /kaggle/working/fl_best_5rounds.pt \
  --out /kaggle/working
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from isic_common import CLASSES, ISICDataset, create_model, load_split


def infer(checkpoint, loader, device):
    net = create_model(pretrained=False)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(bundle["model_state_dict"], strict=True)
    net.to(device).eval()
    ys, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            output = net(images.to(device))
            ys.extend(labels.tolist())
            preds.extend(output.argmax(dim=1).cpu().tolist())
    return ys, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--federated", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    val = load_split(args.data_dir / "isic2019_val.csv")
    loader = DataLoader(ISICDataset(val), batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, reports, raw = {}, {}, {}
    for name, checkpoint in [("centralized", args.central), ("federated", args.federated)]:
        y, ypred = infer(checkpoint, loader, device)
        metrics[name] = {"accuracy": accuracy_score(y, ypred),
                         "macro_f1": f1_score(y, ypred, labels=list(range(8)),
                                              average="macro", zero_division=0),
                         "balanced_accuracy": balanced_accuracy_score(y, ypred)}
        reports[name] = classification_report(y, ypred, labels=list(range(8)),
                                              target_names=CLASSES, output_dict=True,
                                              zero_division=0)
        raw[name] = (y, ypred)
        pd.DataFrame(confusion_matrix(y, ypred, labels=list(range(8))),
                     index=CLASSES, columns=CLASSES).to_csv(args.out / f"{name}_confusion_counts.csv")
    pd.DataFrame(metrics).to_csv(args.out / "model_comparison.csv")
    pd.DataFrame({
        f"{name}_f1": [report[c]["f1-score"] for c in CLASSES]
        for name, report in reports.items()
    }, index=CLASSES).to_csv(args.out / "per_class_f1.csv")
    pd.DataFrame({
        f"{name}_recall": [report[c]["recall"] for c in CLASSES]
        for name, report in reports.items()
    }, index=CLASSES).to_csv(args.out / "per_class_recall.csv")
    print(pd.DataFrame(metrics).round(4).to_string())


if __name__ == "__main__":
    main()
