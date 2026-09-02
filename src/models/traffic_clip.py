import torch
import torch.nn as nn

from src.models.detail_encoder import DetailAwareEncoder
from src.models.semantic_encoder import SemanticEncoder
from src.models.adapter import TrafficAdapter
from src.models.stats_projection import StatsProjectionHead
from src.models.fusion import FusionModule
from src.models.text_encoder import TextEncoder


class TrafficCLIP(nn.Module):

    def __init__(self, embed_dim=1024):

        super().__init__()

        self.detail_encoder = DetailAwareEncoder()

        self.semantic_encoder = SemanticEncoder()

        # Semantic ResNet is a frozen feature extractor.
        for param in self.semantic_encoder.parameters():
            param.requires_grad = False

        self.adapter = TrafficAdapter()

        self.stats_projection = StatsProjectionHead()

        self.fusion = FusionModule(
            detail_dim=512,
            semantic_dim=512,
            stats_dim=512,
            output_dim=embed_dim
        )

        self.text_encoder = TextEncoder(
            output_dim=embed_dim
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) *
            torch.log(torch.tensor(1 / 0.07))
        )

    def encode_visual(
        self,
        images,
        stats
    ):

        # Detail branch
        detail_feat = self.detail_encoder(
            images
        )

        # Frozen semantic branch
        with torch.no_grad():

            semantic_feat = self.semantic_encoder(
                images
            )

        semantic_feat = self.adapter(
            semantic_feat
        )

        # Statistical branch
        stats_feat = self.stats_projection(
            stats
        )

        # Fusion
        visual_feat = self.fusion(
            detail_feat,
            semantic_feat,
            stats_feat
        )

        visual_feat = visual_feat / (
            visual_feat.norm(
                dim=-1,
                keepdim=True
            ) + 1e-8
        )

        return visual_feat

    def forward(
        self,
        images,
        stats,
        texts
    ):

        visual_feat = self.encode_visual(
            images,
            stats
        )

        text_feat = self.text_encoder(
            texts
        )

        text_feat = text_feat / (
            text_feat.norm(
                dim=-1,
                keepdim=True
            ) + 1e-8
        )

        logit_scale = self.logit_scale.exp()

        logits_per_image = (
            logit_scale *
            visual_feat @ text_feat.t()
        )

        logits_per_text = logits_per_image.t()

        return (
            logits_per_image,
            logits_per_text
        )