"""Common data/model functions for a reproducible ISIC 2019 Stage 1 baseline.

Reconstructed from the documented Kaggle experiments. Source scripts are provided
as reproduction aids; the exported original Kaggle notebook is authoritative.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transforms(training=False):
    ops = [transforms.Resize((224, 224))]
    if training:
        ops += [transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class ISICDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, training=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = image_transforms(training=training)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row.image_path) as im:
            image = self.transform(im.convert("RGB"))
        return image, int(row.label_id)


def create_model(pretrained=False, seed=42):
    if pretrained:
        torch.manual_seed(seed)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.layer4.parameters():
        p.requires_grad = True
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    return net


def training_weights(df):
    counts = torch.bincount(torch.as_tensor(df.label_id.values, dtype=torch.long),
                            minlength=len(CLASSES)).float()
    if (counts <= 0).any():
        raise ValueError("Every class must have training samples")
    weights = counts.rsqrt()
    return weights / weights.mean()


def load_split(path):
    df = pd.read_csv(path)
    needed = {"image_path", "label_id", "diagnosis", "group_id"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing CSV columns: {needed - set(df.columns)}")
    missing = sum(not Path(p).is_file() for p in df.image_path)
    if missing:
        raise FileNotFoundError(f"{missing} image paths in {path} do not exist. "
                                "Recreate splits for your current environment.")
    return df
