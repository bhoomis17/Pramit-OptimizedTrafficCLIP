import torch
import torch.nn as nn


class StatsProjectionHead(nn.Module):
    """
    Projects the 3 statistical traffic features
    (mean IAT, jitter, entropy) into a 512-dimensional
    embedding.
    """

    def __init__(self, input_dim=3, hidden_dim=128, output_dim=512):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, stats):
        return self.projection(stats)


if __name__ == "__main__":
    # Simple test
    model = StatsProjectionHead()

    # Example:
    # [mean IAT, jitter, entropy]
    x = torch.tensor(
        [[12.5, 3.2, 7.8]],
        dtype=torch.float32
    )

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)