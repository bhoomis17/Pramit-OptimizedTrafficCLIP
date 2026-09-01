import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrafficDataset(Dataset):

    def __init__(
        self,
        csv_file,
        class_to_idx=None,
        image_size=28,
        statistics_csv=None
    ):
        # Read original CSV containing image_path and label
        self.data = pd.read_csv(csv_file)

        # Read statistics CSV if provided
        if statistics_csv is not None:

            statistics_data = pd.read_csv(statistics_csv)

            # Match using image filename
            statistics_data["image_name"] = (
                statistics_data["image_path"].apply(os.path.basename)
            )

            self.data["image_name"] = (
                self.data["image_path"].apply(os.path.basename)
            )

            # Merge statistics with dataset
            self.data = self.data.merge(
                statistics_data[
                    [
                        "image_name",
                        "mean_iat",
                        "jitter",
                        "entropy"
                    ]
                ],
                on="image_name",
                how="left"
            )

            # Remove helper column
            self.data.drop(
                columns=["image_name"],
                inplace=True
            )

            # Ensure every image got statistics
            if self.data[
                ["mean_iat", "jitter", "entropy"]
            ].isnull().any().any():

                raise ValueError(
                    "Some images do not have corresponding statistical features."
                )

        else:

            # Keep base model compatible
            self.data["mean_iat"] = 0.0
            self.data["jitter"] = 0.0
            self.data["entropy"] = 0.0

        # Class mapping
        if class_to_idx is None:

            classes = sorted(self.data["label"].unique())

            self.class_to_idx = {
                name: idx
                for idx, name in enumerate(classes)
            }

        else:

            self.class_to_idx = class_to_idx

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("L")

        image = self.transform(image)

        label = self.class_to_idx[row["label"]]

        # Create 3-value statistics tensor
        stats = torch.tensor(
            [
                row["mean_iat"],
                row["jitter"],
                row["entropy"]
            ],
            dtype=torch.float32
        )

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": row["label"],
            "stats": stats
        }