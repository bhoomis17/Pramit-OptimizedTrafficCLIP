import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Fuses:
    - Detail features:   512
    - Semantic features: 512
    - Statistical features: 512

    Total input: 1536
    Output: 1024
    """

    def __init__(
        self,
        detail_dim=512,
        semantic_dim=512,
        stats_dim=512,
        output_dim=1024
    ):
        super().__init__()

        combined_dim = (
            detail_dim
            + semantic_dim
            + stats_dim
        )

        self.proj = nn.Linear(
            combined_dim,
            output_dim
        )

    def forward(
        self,
        detail_feat,
        semantic_feat,
        stats_feat
    ):
        fused = torch.cat(
            [
                detail_feat,
                semantic_feat,
                stats_feat
            ],
            dim=1
        )

        return self.proj(fused)