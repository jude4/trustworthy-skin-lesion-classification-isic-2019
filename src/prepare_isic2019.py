"""Create lesion-aware splits and 3 group-preserving non-IID client partitions.

Example (from the repository root):
python src/prepare_isic2019.py \
  --images-dir /kaggle/input/datasets/nischaydnk/isic-2019-jpg-224x224-resized/train-image/image \
  --labels /kaggle/working/ISIC_2019_Training_GroundTruth.csv \
  --metadata /kaggle/working/ISIC_2019_Training_Metadata.csv \
  --out /kaggle/working
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from isic_common import CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels)
    metadata = pd.read_csv(args.metadata)
    if "UNK" in labels:
        assert (labels.UNK.fillna(0) == 0).all(), "Non-zero UNK examples need handling"
    assert (labels[CLASSES].sum(axis=1) == 1).all()
    df = labels.merge(metadata, on="image", validate="one_to_one", how="left")
    df["diagnosis"] = df[CLASSES].idxmax(axis=1)
    df["label_id"] = pd.Categorical(df.diagnosis, categories=CLASSES).codes
    df["image_path"] = df.image.apply(lambda x: str(args.images_dir / (x + ".jpg")))
    df["group_id"] = df.lesion_id.fillna("image_" + df.image)
    if not all(Path(p).is_file() for p in df.image_path):
        raise FileNotFoundError("Image files do not match the official label IDs")
    df.to_csv(args.out / "isic2019_verified.csv", index=False)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    itrainval, itest = next(sgkf.split(df, df.label_id, df.group_id))
    trainval, test = df.iloc[itrainval].copy(), df.iloc[itest].copy()
    itr, ival = next(sgkf.split(trainval, trainval.label_id, trainval.group_id))
    train, val = trainval.iloc[itr].copy(), trainval.iloc[ival].copy()
    frames = {"train": train, "val": val, "test": test}
    groups = {k: set(v.group_id) for k, v in frames.items()}
    assert not (groups["train"] & groups["val"] or
                groups["train"] & groups["test"] or
                groups["val"] & groups["test"])
    for name, frame in frames.items():
        frame.to_csv(args.out / f"isic2019_{name}.csv", index=False)
        print(name, len(frame), frame.diagnosis.value_counts().to_dict())

    # Same group-aware Dirichlet partition used in the reported Kaggle experiment.
    rng = np.random.default_rng(args.seed)
    group_info = (train.groupby("group_id").diagnosis
                  .agg(lambda x: x.value_counts().idxmax()).reset_index())
    for attempt in range(100):
        assignment = {}
        for diagnosis in CLASSES:
            g = group_info.loc[group_info.diagnosis == diagnosis, "group_id"].to_numpy()
            rng.shuffle(g)
            proportions = rng.dirichlet(np.full(args.num_clients, args.alpha))
            chunks = np.split(g, np.cumsum(rng.multinomial(len(g), proportions))[:-1])
            for client_id, chunk in enumerate(chunks):
                for group in chunk:
                    assignment[group] = client_id
        ids = train.group_id.map(assignment)
        sizes = ids.value_counts().reindex(range(args.num_clients), fill_value=0)
        if sizes.min() >= 1000:
            break
    else:
        raise RuntimeError("Could not form sufficiently sized partitions")
    train["client_id"] = ids.astype(int)
    assert (train.groupby("group_id").client_id.nunique() == 1).all()
    train.to_csv(args.out / "isic2019_federated_partitions.csv", index=False)
    print(pd.crosstab(train.client_id, train.diagnosis).reindex(columns=CLASSES, fill_value=0))
    print("Saved all split and partition CSV files into", args.out)


if __name__ == "__main__":
    main()
