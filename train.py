import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.datasets.traffic_dataset import TrafficDataset
from src.training.trainer import Trainer


def make_dummy_dataset(n_per_class=8):
    """
    Generates placeholder images so the complete training pipeline
    can be tested before real preprocessed images are available.

    The dummy dataset does not contain real traffic statistics, so
    TrafficDataset will use zero values for:
        - mean_iat
        - jitter
        - entropy
    """

    tmp_dir = tempfile.mkdtemp()

    csv_path = os.path.join(
        tmp_dir,
        "dummy_train.csv"
    )

    rows = []

    classes = [
        "Skype",
        "Zoom",
        "BitTorrent"
    ]

    for c in classes:

        for i in range(n_per_class):

            img_path = os.path.join(
                tmp_dir,
                f"{c}_{i}.png"
            )

            # Create random 28x28 grayscale image
            arr = np.random.randint(
                0,
                255,
                (28, 28),
                dtype=np.uint8
            )

            PILImage.fromarray(
                arr,
                mode="L"
            ).save(img_path)

            rows.append(
                (
                    img_path,
                    c
                )
            )

    pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "label"
        ]
    ).to_csv(
        csv_path,
        index=False
    )

    return csv_path


def main():

    # ============================================================
    # Command-line arguments
    # ============================================================

    parser = argparse.ArgumentParser(
        description="Train TrafficCLIP"
    )

    parser.add_argument(
        "--train_csv",
        default=None,
        help=(
            "Path to training CSV containing "
            "image_path and label"
        )
    )

    parser.add_argument(
        "--statistics_csv",
        default=None,
        help=(
            "Path to statistics CSV containing "
            "image_path, mean_iat, jitter and entropy"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="Learning rate"
    )

    args = parser.parse_args()

    # ============================================================
    # Determine whether real training data was supplied
    # ============================================================

    use_dummy_data = (
        args.train_csv is None
        or not os.path.exists(args.train_csv)
        or os.path.getsize(args.train_csv) < 30
    )

    if use_dummy_data:

        print(
            "No real training data found — "
            "using dummy generated images "
            "for a pipeline smoke test."
        )

        csv_path = make_dummy_dataset()

        # Dummy data has no real statistics
        statistics_csv = None

    else:

        csv_path = args.train_csv

        # ========================================================
        # Validate statistics CSV when real training data is used
        # ========================================================

        if args.statistics_csv is None:

            raise ValueError(
                "Real training data was supplied, but "
                "--statistics_csv was not provided.\n\n"
                "TrafficCLIP now requires the statistical "
                "features:\n"
                "  mean_iat\n"
                "  jitter\n"
                "  entropy\n\n"
                "Please provide --statistics_csv."
            )

        if not os.path.exists(
            args.statistics_csv
        ):

            raise FileNotFoundError(
                "Statistics CSV not found:\n"
                f"{args.statistics_csv}"
            )

        if os.path.getsize(
            args.statistics_csv
        ) < 30:

            raise ValueError(
                "The supplied statistics CSV appears "
                "to be empty or invalid."
            )

        statistics_csv = args.statistics_csv

    # ============================================================
    # Load dataset
    # ============================================================

    dataset = TrafficDataset(
        csv_file=csv_path,
        statistics_csv=statistics_csv
    )

    # ============================================================
    # Get class names
    # ============================================================

    class_names = list(
        dataset.class_to_idx.keys()
    )

    print(
        f"Classes: {class_names}"
    )

    print(
        f"Number of training samples: {len(dataset)}"
    )

    # ============================================================
    # Display statistics information
    # ============================================================

    if statistics_csv is not None:

        print(
            f"Statistics CSV: {statistics_csv}"
        )

        print(
            "Statistical features: "
            "mean_iat, jitter, entropy"
        )

    else:

        print(
            "Statistics CSV: None"
        )

        print(
            "Using zero statistical features "
            "for dummy smoke test."
        )

    # ============================================================
    # Create Trainer
    # ============================================================

    trainer = Trainer(
        dataset=dataset,
        class_names=class_names,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )

    # ============================================================
    # Train TrafficCLIP
    # ============================================================

    losses = trainer.train()

    # ============================================================
    # Save training results
    # ============================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    results_path = os.path.join(
        "outputs",
        "results.txt"
    )

    # Calculate number of batches in the final epoch
    final_epoch_batches = len(
        trainer.dataloader
    )

    if final_epoch_batches > 0:

        final_epoch_losses = losses[
            -final_epoch_batches:
        ]

        average_final_epoch_loss = (
            sum(final_epoch_losses)
            / len(final_epoch_losses)
        )

    else:

        average_final_epoch_loss = 0.0

    with open(
        results_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n--- train.py run ---\n"
        )

        f.write(
            f"Training CSV: {csv_path}\n"
        )

        f.write(
            f"Statistics CSV: {statistics_csv}\n"
        )

        f.write(
            f"Classes: {class_names}\n"
        )

        f.write(
            f"Number of samples: {len(dataset)}\n"
        )

        f.write(
            f"Epochs: {args.epochs}\n"
        )

        f.write(
            f"Batch size: {args.batch_size}\n"
        )

        f.write(
            f"Learning rate: {args.lr}\n"
        )

        f.write(
            f"Final losses per step: {losses}\n"
        )

        f.write(
            "Average final-epoch loss: "
            f"{average_final_epoch_loss:.4f}\n"
        )

    # ============================================================
    # Final message
    # ============================================================

    print(
        "Training complete."
    )

    print(
        f"Results saved to: {results_path}"
    )

    print(
        "Checkpoint saved to: "
        "outputs/traffic_clip_best.pt"
    )


if __name__ == "__main__":
    main()