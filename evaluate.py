import argparse
import os
import tempfile
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from src.datasets.traffic_dataset import TrafficDataset
from src.models.traffic_clip import TrafficCLIP
from src.utils.metrics import compute_metrics


def make_dummy_dataset(n_per_class=4):
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "dummy_eval.csv")
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
    parser = argparse.ArgumentParser(description="Evaluate TrafficCLIP")
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.test_csv is None or not os.path.exists(args.test_csv) or os.path.getsize(args.test_csv) < 30:
        print("No real test data found — using dummy generated images for a pipeline smoke test.")
        csv_path = make_dummy_dataset()
    else:
        csv_path = args.test_csv

    dataset = TrafficDataset(csv_path)
    class_names = list(dataset.class_to_idx.keys())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrafficCLIP(embed_dim=1024).to(device)
    model.eval()

    text_prompts = [f"a network traffic photo of {c}" for c in class_names]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits_per_image, _ = model(images, text_prompts)
            preds = logits_per_image.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.txt", "a") as f:
        f.write(f"\n--- evaluate.py run ---\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}\n")

    print("Evaluation complete. Results appended to outputs/results.txt")


if __name__ == "__main__":
    main()