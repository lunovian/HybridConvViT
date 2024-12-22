import torch
import torch.nn as nn

class CocaEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_heads, num_layers):
        super(CocaEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Transformer blocks (simplified example)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.conv_layers(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return x