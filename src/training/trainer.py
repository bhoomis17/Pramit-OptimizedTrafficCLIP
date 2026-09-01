import torch
from torch.utils.data import DataLoader

from src.models.traffic_clip import TrafficCLIP
from src.losses.cross_entropy import TrafficCrossEntropy
from src.losses.contrastive import SupConLoss
from src.training.scheduler import get_scheduler
from src.utils.seed import set_seed


class Trainer:
    def __init__(self, dataset, class_names, batch_size=4, lr=0.002, epochs=2, device=None):
        set_seed(42)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.class_names = class_names
        self.epochs = epochs

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = TrafficCLIP(embed_dim=1024).to(self.device)
        self.ce_loss = TrafficCrossEntropy()
        self.cl_loss = SupConLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = get_scheduler(self.optimizer, total_epochs=epochs, base_lr=lr)

        # Prompt template from the paper: "a network traffic photo of {}"
        self.text_prompts = [f"a network traffic photo of {c}" for c in class_names]

    def train(self):
        self.model.train()
        losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits_per_image, _ = self.model(images, self.text_prompts)

                loss_ce = self.ce_loss(logits_per_image, labels)

                # Contrastive loss needs the fused visual features, not the final logits.
                # We recompute them the same way the model does internally.
                detail_feat = self.model.detail_encoder(images)
                semantic_feat = self.model.adapter(self.model.semantic_encoder(images))
                visual_feat = self.model.fusion(detail_feat, semantic_feat)
                loss_cl = self.cl_loss(visual_feat, labels)

                loss = loss_ce + loss_cl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                losses.append(loss.item())

            self.scheduler.step()
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return losses


if __name__ == "__main__":
    # Quick self-test with dummy data, same pattern as traffic_dataset.py
    import tempfile
    import os
    import pandas as pd
    import numpy as np
    from PIL import Image as PILImage
    from src.datasets.traffic_dataset import TrafficDataset

    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "dummy.csv")
    rows = []
    for i, label in enumerate(["Skype", "Zoom"] * 2):
        img_path = os.path.join(tmp_dir, f"img_{i}.png")
        arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        PILImage.fromarray(arr, mode="L").save(img_path)
        rows.append((img_path, label))
    pd.DataFrame(rows, columns=["image_path", "label"]).to_csv(csv_path, index=False)

    dataset = TrafficDataset(csv_path)
    class_names = list(dataset.class_to_idx.keys())

    trainer = Trainer(dataset, class_names, batch_size=2, epochs=2)
    losses = trainer.train()
    print(f"Final losses: {losses}")