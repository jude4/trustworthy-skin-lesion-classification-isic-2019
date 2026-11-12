"""Flower 1.43 FedAvg simulation for ISIC 2019 Stage 1.

Example on Kaggle:
ISIC_DATA_DIR=/kaggle/working ISIC_OUTPUT_DIR=/kaggle/working \
ISIC_FL_ROUNDS=5 python src/train_federated.py

This is a reconstruction of the working experiment code shared in conversation.
The original, exported Kaggle training notebook/script should be submitted too.
"""
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation

from isic_common import (CLASSES, ISICDataset, create_model,
                         load_split, training_weights)

DATA_DIR = Path(os.environ.get("ISIC_DATA_DIR", "/kaggle/working"))
OUT_DIR = Path(os.environ.get("ISIC_OUTPUT_DIR", "/kaggle/working"))
NUM_CLIENTS = 3
NUM_ROUNDS = int(os.environ.get("ISIC_FL_ROUNDS", "5"))
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
OUT_DIR.mkdir(parents=True, exist_ok=True)


def floating_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            if torch.is_floating_point(v)}


def load_floating_state(net, state):
    weights = net.state_dict()
    weights.update(state)
    net.load_state_dict(weights)


def train_local(net, loader, weights, device):
    net.to(device)
    net.train()
    for block in (net.bn1, net.layer1, net.layer2, net.layer3):
        block.eval()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                  lr=LEARNING_RATE, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    loss_total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = net(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total += float(loss.item()) * len(images)
    return loss_total / len(loader.dataset)


client_app = ClientApp()


@client_app.train()
def client_train(msg: Message, context: Context):
    torch.set_num_threads(2)
    client_id = int(context.node_config["partition-id"])
    data = load_split(DATA_DIR / "isic2019_federated_partitions.csv")
    local_data = data.loc[data.client_id == client_id]
    loader = DataLoader(ISICDataset(local_data, training=True), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    net = create_model(pretrained=False)
    load_floating_state(net, msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client {client_id + 1}: {len(local_data)} images, device={device}", flush=True)
    weights = training_weights(load_split(DATA_DIR / "isic2019_train.csv"))
    loss = train_local(net, loader, weights, device)
    return Message(content=RecordDict({
        "arrays": ArrayRecord(floating_state(net)),
        "metrics": MetricRecord({"num-examples": len(local_data), "train_loss": float(loss)})
    }), reply_to=msg)


def evaluate_global(net):
    torch.set_num_threads(2)
    net.eval()
    data = load_split(DATA_DIR / "isic2019_val.csv")
    loader = DataLoader(ISICDataset(data), batch_size=32, shuffle=False, num_workers=0)
    weights = training_weights(load_split(DATA_DIR / "isic2019_train.csv"))
    criterion = nn.CrossEntropyLoss(weight=weights)
    ys, preds, loss_total = [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            outputs = net(images)
            loss_total += float(criterion(outputs, labels).item()) * len(images)
            ys.extend(labels.tolist())
            preds.extend(outputs.argmax(dim=1).tolist())
    return {"val_loss": loss_total / len(data),
            "val_macro_f1": float(f1_score(ys, preds, labels=list(range(8)),
                                            average="macro", zero_division=0)),
            "val_balanced_acc": float(balanced_accuracy_score(ys, preds))}


server_app = ServerApp()


@server_app.main()
def server_main(grid, context):
    global_net = create_model(pretrained=True)
    best_f1, history = -1.0, []

    def evaluate_round(round_number, arrays):
        nonlocal best_f1
        net = create_model(pretrained=False)
        load_floating_state(net, arrays.to_torch_state_dict())
        metrics = evaluate_global(net)
        print(f"Round {round_number}: {metrics}", flush=True)
        history.append({"round": round_number, **metrics})
        (OUT_DIR / f"fl_history_{NUM_ROUNDS}rounds.json").write_text(
            json.dumps(history, indent=2))
        if round_number > 0 and metrics["val_macro_f1"] > best_f1:
            best_f1 = metrics["val_macro_f1"]
            torch.save({"model_state_dict": net.state_dict(),
                        "round": round_number, "val_macro_f1": best_f1,
                        "classes": CLASSES}, OUT_DIR / f"fl_best_{NUM_ROUNDS}rounds.pt")
        return MetricRecord(metrics)

    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=0.0,
                      min_train_nodes=NUM_CLIENTS,
                      min_evaluate_nodes=0, min_available_nodes=NUM_CLIENTS)
    strategy.start(grid=grid, initial_arrays=ArrayRecord(floating_state(global_net)),
                   num_rounds=NUM_ROUNDS,
                   train_config=ConfigRecord({"lr": LEARNING_RATE}),
                   evaluate_fn=evaluate_round)
    print(f"Federated training completed. Best validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    if not (DATA_DIR / "isic2019_federated_partitions.csv").exists():
        raise FileNotFoundError("First run prepare_isic2019.py or export your Kaggle partitions")
    # 1 simulated client per GPU; adjust for local CPU-only environments.
    use_gpu = torch.cuda.is_available()
    run_simulation(server_app=server_app, client_app=client_app,
                   num_supernodes=NUM_CLIENTS,
                   backend_config={"client_resources": {
                       "num_cpus": 1, "num_gpus": 1.0 if use_gpu else 0.0}})
