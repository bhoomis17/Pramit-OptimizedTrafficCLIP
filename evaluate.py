import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.datasets.traffic_dataset import TrafficDataset
from src.models.traffic_clip import TrafficCLIP

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TrafficCLIP on the real test set")
    parser.add_argument("--test_csv", required=True, help="Path to real test CSV")
    parser.add_argument("--checkpoint", default="outputs/traffic_clip_best.pt", help="Path to trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    args = parser.parse_args()

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    print(f"Classes: {class_names}")

    class_to_idx = {name: index for index, name in enumerate(class_names)}
    dataset = TrafficDataset(csv_file=args.test_csv, class_to_idx=class_to_idx)
    print(f"Number of test samples: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading TrafficCLIP...")
    model = TrafficCLIP(embed_dim=1024).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Checkpoint loaded successfully.")

    text_prompts = [f"a network traffic photo of {c}" for c in class_names]
    all_labels = []
    all_preds = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            stats = batch["stats"].to(device)
            logits_per_image, _ = model(images, stats, text_prompts)
            preds = logits_per_image.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(dataloader):
                print(f"  Batch {batch_idx}/{len(dataloader)}")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)

    print()
    print("=" * 70)
    print("TrafficCLIP TEST RESULTS")
    print("=" * 70)
    print(f"Test samples    : {len(dataset)}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1        : {f1:.4f}")
    print()
    print("Classification Report")
    print("=" * 70)
    print(report)
    print()
    print("10-Class Confusion Matrix")
    print("=" * 70)
    print("Rows = Actual")
    print("Columns = Predicted")
    print()
    print("             " + " ".join(f"{i:5d}" for i in range(len(class_names))))
    for i, row in enumerate(cm):
        print(f"{class_names[i][:11]:12s}" + " ".join(f"{value:5d}" for value in row))

    os.makedirs("outputs", exist_ok=True)
    results_path = "outputs/evaluation_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("TrafficCLIP TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test samples: {len(dataset)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write("Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(report)
        f.write("\nConfusion Matrix\n")
        f.write("=" * 70 + "\n")
        f.write(np.array2string(cm))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("TrafficCLIP Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    confusion_path = "outputs/confusion_matrix.png"
    plt.savefig(confusion_path, dpi=200)
    plt.close()

    print()
    print("=" * 70)
    print("Evaluation complete.")
    print(f"Results saved to: {results_path}")
    print(f"Confusion matrix saved to: {confusion_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
