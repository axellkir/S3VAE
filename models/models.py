import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist


class Encoder(nn.Module):
    def __init__(self, image_shape=3):
        super(Encoder, self).__init__()
        self.image_shape = image_shape
        self.main_net = nn.Sequential(
            nn.Conv2d(in_channels=self.image_shape, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=4, stride=1, padding=0),
        )
        # self.batch_norm = nn.BatchNorm1d(num_features=128)

    def forward(self, batch):
        shape, feature_shape = batch.shape[:-3], batch.shape[-3:]
        batch = batch.reshape(-1, *feature_shape)
        encoded = self.main_net(batch)
        encoded = encoded.reshape(*shape, -1)
        return F.elu(encoded)


class Decoder(nn.Module):
    def __init__(self, input_size=384,  image_shape=3):
        super().__init__()
        self.image_shape = image_shape

        self.main_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_size, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.image_shape, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, batch):
        shape, feature_shape = batch.shape[:-1], batch.shape[-1]
        batch = batch.reshape(-1, feature_shape, 1, 1)
        decoded = self.main_net(batch)
        feature_shape = decoded.shape[1:]
        decoded = decoded.reshape(*shape, *feature_shape)
        return dist.Normal(loc=decoded, scale=1)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=128, z_size=128, static=False):
        super().__init__()
        self.main_net = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1)
        self.mean = nn.Linear(in_features=256 * (1 + int(static)), out_features=z_size)
        self.std = nn.Linear(in_features=256 * (1 + int(static)), out_features=z_size)
        self.static = static

    def forward(self, encoded):
        outs, hidden = self.main_net(encoded)
        if self.static:
            hidden = torch.cat(hidden, 2).squeeze(0)
            mean = self.mean(hidden)
            std = F.softplus(self.std(hidden))
        else:
            mean = self.mean(outs)
            std = F.softplus(self.std(outs))
        return dist.Normal(loc=mean, scale=std)


class DynamicFactorPrediction(nn.Module):
    def __init__(self, z_size=128):
        super().__init__()
        self.main_net = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=z_size),
            nn.Linear(in_features=z_size, out_features=z_size)
        )
        self.mean = nn.Linear(in_features=z_size, out_features=3)
        self.std = nn.Linear(in_features=z_size, out_features=3)

    def forward(self, batch):
        shape, feature_shape = batch.shape[:-1], batch.shape[-1]
        batch = batch.reshape(-1, feature_shape)
        features = self.main_net(batch)
        feature_shape = features.shape[1:]
        features = features.reshape(*shape, *feature_shape)
        # features = self.main_net(z_t)
        mean = self.mean(features)
        std = F.softplus(self.std(features))
        return dist.Normal(loc=mean, scale=std)

