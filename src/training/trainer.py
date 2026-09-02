import os

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from src.models.traffic_clip import TrafficCLIP
from src.losses.cross_entropy import TrafficCrossEntropy
from src.losses.contrastive import SupConLoss
from src.training.scheduler import get_scheduler
from src.utils.seed import set_seed


class Trainer:

    def __init__(
        self,
        dataset,
        class_names,
        batch_size=4,
        lr=0.0001,
        epochs=2,
        device=None
    ):

        set_seed(42)

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Device: {self.device}")

        self.dataset = dataset
        self.class_names = class_names
        self.epochs = epochs

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        print("Loading TrafficCLIP model...")

        self.model = TrafficCLIP(
            embed_dim=1024
        ).to(self.device)

        self.ce_loss = TrafficCrossEntropy()

        self.cl_loss = SupConLoss()

        # Only trainable parameters are optimized.
        trainable_parameters = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]

        print(
            "Trainable parameters:",
            sum(
                p.numel()
                for p in trainable_parameters
            )
        )

        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=lr,
            weight_decay=1e-4
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            total_epochs=epochs,
            base_lr=lr
        )

        self.text_prompts = [
            f"a network traffic photo of {c}"
            for c in class_names
        ]

        print("TrafficCLIP ready.")
        print(
            f"Training batches per epoch: "
            f"{len(self.dataloader)}"
        )

    def train(self):

        self.model.train()

        # Keep frozen semantic encoder in evaluation mode.
        self.model.semantic_encoder.eval()

        # Keep frozen BERT in evaluation mode.
        self.model.text_encoder.bert.eval()

        losses = []

        total_batches = len(
            self.dataloader
        )

        for epoch in range(self.epochs):

            epoch_loss = 0.0

            print(
                f"\nEpoch {epoch + 1}/{self.epochs}"
            )

            for batch_idx, batch in enumerate(
                self.dataloader,
                start=1
            ):

                images = batch["image"].to(
                    self.device
                )

                labels = batch["label"].to(
                    self.device
                )

                stats = batch["stats"].to(
                    self.device
                )

                # ------------------------------------------------
                # ONE visual forward pass
                # ------------------------------------------------

                visual_feat = self.model.encode_visual(
                    images,
                    stats
                )

                # ------------------------------------------------
                # Text encoding
                # BERT itself is frozen.
                # Only the 768 -> 1024 projection is trainable.
                # ------------------------------------------------

                text_feat = self.model.text_encoder(
                    self.text_prompts
                )

                text_feat = text_feat / (
                    text_feat.norm(
                        dim=-1,
                        keepdim=True
                    ) + 1e-8
                )

                logit_scale = (
                    self.model.logit_scale.exp()
                )

                logits_per_image = (
                    logit_scale *
                    visual_feat @ text_feat.t()
                )

                # ------------------------------------------------
                # Classification loss
                # ------------------------------------------------

                loss_ce = self.ce_loss(
                    logits_per_image,
                    labels
                )

                # ------------------------------------------------
                # Supervised contrastive loss
                # ------------------------------------------------

                loss_cl = self.cl_loss(
                    visual_feat,
                    labels
                )

                # ------------------------------------------------
                # Total loss
                # ------------------------------------------------

                loss = (
                    loss_ce +
                    loss_cl
                )

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                self.optimizer.step()

                # ------------------------------------------------
                # Record
                # ------------------------------------------------

                loss_value = loss.item()

                epoch_loss += loss_value

                losses.append(
                    loss_value
                )

                # Print progress every 25 batches
                if (
                    batch_idx == 1
                    or batch_idx % 25 == 0
                    or batch_idx == total_batches
                ):

                    print(
                        f"  Batch "
                        f"{batch_idx}/{total_batches} "
                        f"- Loss: {loss_value:.4f}",
                        flush=True
                    )

            self.scheduler.step()

            avg_loss = (
                epoch_loss /
                total_batches
            )

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"completed - "
                f"Average Loss: {avg_loss:.4f}",
                flush=True
            )

        # --------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        checkpoint_path = (
            "outputs/traffic_clip_best.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "class_names":
                    self.class_names,

                "epochs":
                    self.epochs,
            },
            checkpoint_path
        )

        print(
            f"\nModel checkpoint saved to: "
            f"{checkpoint_path}"
        )

        return losses


if __name__ == "__main__":

    print(
        "Trainer module loaded successfully."
    )