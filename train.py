import argparse
import os
import tempfile
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.datasets.traffic_dataset import TrafficDataset
from src.training.trainer import Trainer


def make_dummy_dataset(n_per_class=8):
    """Generates placeholder data so the pipeline can run before real
    preprocessed images exist. Swap --train_csv to a real CSV once
    preprocess.py has produced actual images."""
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "dummy_train.csv")
    rows = []
    classes = ["Skype", "Zoom", "BitTorrent"]
    for c in classes:
        for i in range(n_per_class):
            img_path = os.path.join(tmp_dir, f"{c}_{i}.png")
            arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
            PILImage.fromarray(arr, mode="L").save(img_path)
            rows.append((img_path, c))
    pd.DataFrame(rows, columns=["image_path", "label"]).to_csv(csv_path, index=False)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Train TrafficCLIP")
    parser.add_argument("--train_csv", default=None, help="Path to train.csv (image_path,label)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.train_csv is None or not os.path.exists(args.train_csv) or os.path.getsize(args.train_csv) < 30:
        print("No real training data found — using dummy generated images for a pipeline smoke test.")
        csv_path = make_dummy_dataset()
    else:
        csv_path = args.train_csv

    dataset = TrafficDataset(csv_path)
    class_names = list(dataset.class_to_idx.keys())
    print(f"Classes: {class_names}")

    trainer = Trainer(dataset, class_names, batch_size=args.batch_size, epochs=args.epochs)
    losses = trainer.train()

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.txt", "a") as f:
        f.write(f"\n--- train.py run ---\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Final losses per step: {losses}\n")
        f.write(f"Average final-epoch loss: {sum(losses[-len(dataset)//args.batch_size:]) / max(1, len(dataset)//args.batch_size):.4f}\n")

    print("Training complete. Results appended to outputs/results.txt")


if __name__ == "__main__":
    main()