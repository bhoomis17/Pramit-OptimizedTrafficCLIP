import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def numbers_after_label(text, label):
    m = re.search(rf"{label}\\s*[:=]\\s*([0-9.]+)", text, re.I)
    if m:
        v = float(m.group(1))
        return v * 100 if v <= 1 else v
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="outputs")
    args = p.parse_args()

    out = Path(args.outputs_dir)
    ev = (out / "evaluation_results.txt").read_text(encoding="utf-8", errors="ignore")
    res = (out / "results.txt").read_text(encoding="utf-8", errors="ignore")
    graph = out / "graphs"
    graph.mkdir(parents=True, exist_ok=True)

    # 1. Overall test metrics
    acc = numbers_after_label(ev, "Accuracy")
    f1 = numbers_after_label(ev, "Macro F1")
    if acc is not None and f1 is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        vals = [acc, f1]
        bars = ax.bar(["Accuracy", "Macro F1"], vals)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("TrafficCLIP Test Performance")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.2f}%", ha="center")
        fig.tight_layout()
        fig.savefig(graph/"test_metrics.png", dpi=300)
        plt.close(fig)

    # 2. Per-class precision / recall / F1 from sklearn report
    rows = []
    for line in ev.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+(\\d+)\\s*$", line.strip())
        if m:
            rows.append((m.group(1), float(m.group(2))*100, float(m.group(3))*100,
                         float(m.group(4))*100))
    if rows:
        x = np.arange(len(rows))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(rows)*1.1), 7))
        ax.bar(x-w, [r[1] for r in rows], w, label="Precision")
        ax.bar(x,   [r[2] for r in rows], w, label="Recall")
        ax.bar(x+w, [r[3] for r in rows], w, label="F1 Score")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Traffic Class")
        ax.set_title("Per-Class Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph/"class_metrics.png", dpi=300)
        plt.close(fig)

    # 3. Training loss from results.txt
    losses = []
    for line in res.splitlines():
        if "epoch" in line.lower() and "loss" in line.lower():
            nums = re.findall(r"(?<!\\d)(?:\\d+\\.\\d+|\\d+)(?!\\d)", line)
            if len(nums) >= 2:
                losses.append((int(float(nums[0])), float(nums[-1])))
    if losses:
        d = dict(losses)
        epochs = sorted(d)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, [d[e] for e in epochs], marker="o", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Training Loss")
        ax.set_title("TrafficCLIP Training Loss")
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(graph/"training_loss.png", dpi=300)
        plt.close(fig)

    # 4. Confusion matrix: parse rows printed in evaluation_results.txt
    cm_rows = []
    for line in ev.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)\\s+((?:\\d+\\s+)+\\d+)\\s*$", line.strip())
        if m:
            vals = [int(x) for x in m.group(2).split()]
            cm_rows.append((m.group(1), vals))
    if cm_rows and len(cm_rows) == len(cm_rows[0][1]) and all(len(r[1]) == len(cm_rows) for r in cm_rows):
        labels = [r[0] for r in cm_rows]
        cm = np.array([r[1] for r in cm_rows])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("TrafficCLIP Confusion Matrix")
        fig.tight_layout()
        fig.savefig(graph/"confusion_matrix.png", dpi=300)
        plt.close(fig)

    print(f"Graphs saved to: {graph.resolve()}")
    for f in graph.glob("*.png"):
        print(" -", f.name)

if __name__ == "__main__":
    main()
