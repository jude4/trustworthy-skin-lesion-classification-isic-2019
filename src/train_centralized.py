"""Centralized ResNet18 baseline, 5 epochs; saves best validation-macro-F1 checkpoint.

Example: python src/train_centralized.py --data-dir /kaggle/working --out /kaggle/working
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

from isic_common import CLASSES, ISICDataset, create_model, load_split, training_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train = load_split(args.data_dir / "isic2019_train.csv")
    val = load_split(args.data_dir / "isic2019_val.csv")
    train_loader = DataLoader(ISICDataset(train, training=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(ISICDataset(val), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_model(pretrained=True, seed=args.seed).to(device)
    criterion = nn.CrossEntropyLoss(weight=training_weights(train).to(device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                  lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        for block in (net.bn1, net.layer1, net.layer2, net.layer3):
            block.eval()
        running = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(net(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(images)

        net.eval()
        val_loss_sum, ys, preds = 0.0, [], []
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    out = net(images)
                    loss = criterion(out, labels)
                val_loss_sum += float(loss.item()) * len(images)
                ys.extend(labels.cpu().tolist())
                preds.extend(out.argmax(dim=1).cpu().tolist())
        f1 = f1_score(ys, preds, labels=list(range(8)), average="macro", zero_division=0)
        balanced = balanced_accuracy_score(ys, preds)
        record = {"epoch": epoch, "train_loss": running / len(train),
                  "val_loss": val_loss_sum / len(val), "val_macro_f1": float(f1),
                  "val_balanced_acc": float(balanced)}
        history.append(record)
        print(record, flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state_dict": net.state_dict(), "epoch": epoch,
                        "val_macro_f1": float(f1), "classes": CLASSES},
                       args.out / "resnet18_centralized_best.pt")
        (args.out / "centralized_history.json").write_text(json.dumps(history, indent=2))
    print("Best validation macro F1:", best_f1)


if __name__ == "__main__":
    main()
