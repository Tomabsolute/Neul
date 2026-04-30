import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    """Small conditional DCGAN generator for 28x28 grayscale images."""

    def __init__(self, z_dim: int = 100, num_classes: int = 10, embed_dim: int = 50):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_emb(labels)
        x = torch.cat([noise, label_vec], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """Small conditional discriminator using a label image channel."""

    def __init__(self, num_classes: int = 10, embed_dim: int = 28 * 28):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_emb(labels).view(labels.size(0), 1, 28, 28)
        x = torch.cat([images, label_map], dim=1)
        return self.net(x).view(-1)


def init_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
