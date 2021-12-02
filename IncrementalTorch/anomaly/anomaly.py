from typing import Type

from river import base
from river.base import anomaly
import torch
import inspect
import numpy as np
from torch import nn
import pandas as pd

from ..utils import get_optimizer_fn, get_loss_fn, prep_input
from ..nn_functions.anomaly import get_fc_autoencoder


class Autoencoder(base.AnomalyDetector, nn.Module):
    def __init__(
        self,
        loss_fn="mse",
        optimizer_fn: Type[torch.optim.Optimizer] = "adam_w",
        build_fn=None,
        momentum_scaling=0.99,
        device="cpu",
        **net_params,
    ):
        super().__init__()
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optimizer_fn(optimizer_fn)
        self.build_fn = build_fn
        self.net_params = net_params
        self.device = device
        self.scaler = None if not momentum_scaling else ScoreStandardizer(momentum_scaling) 

        self.encoder = None
        self.decoder = None
        self.is_initialized = False

    def learn_one(self, x):
        return self._learn(x)

    def _learn(self, x):
        prep_input(x, device=self.device)

        if self.is_initialized is False:
            self._init_net(n_features=x.shape[1])

        self.train()
        x_pred = self(x)
        loss = self.loss(x_pred, x)

        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self

    def score_one(self, x: dict):
        x = prep_input(x, device=self.device)

        if self.is_initialized is False:
            self._init_net(n_features=x.shape[1])

        self.eval()
        x_rec = self(x)
        loss = self.loss_fn(x_rec, x).item()
        score = loss if self.scaler is None else self.scaler.learn_transform_one(loss)
        return score

    def learn_many(self, x: pd.DataFrame):
        return self._learn(x)

    def score_many(self, x: pd.DataFrame) -> float:
        x = prep_input(x, device=self.device)

        if self.is_initialized is False:
            self._init_net(n_features=x.shape[1])

        self.eval()
        x_rec = self(x)
        loss = torch.mean(
            self.loss_fn(x_rec, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        loss = loss.cpu().detach().numpy()
        score = loss if self.scaler is None else self.scaler.learn_transform_one(loss)
        return score

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _init_net(self, n_features):
        if self.build_fn is None:
            self.build_fn = get_fc_autoencoder

        self.encoder, self.decoder = self.build_fn(
            n_features=n_features, **self._filter_args(self.build_fn)
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            nn.ModuleList([self.encoder, self.decoder]).parameters(),
            **self._filter_args(self.optimizer_fn),
        )
        return optimizer


class AdaptiveAutoencoder(Autoencoder):
    def __init__(
        self,
        loss_fn="mse",
        optimizer_fn: Type[torch.optim.Optimizer] = "adam_w",
        build_fn=None,
        beta=0.99,
        s=0.2,
        momentum_scaling=0.99,
        device="cpu",
        **net_params,
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            momentum_scaling=momentum_scaling,
            **net_params,
        )
        self.beta_scalar = beta
        self.dropout = None
        self.s = s

    def compute_recs(self, x, train=True):
        x_encs = []
        x_recs = []

        if train and self.dropout is not None:
            x_enc_prev = self.dropout(x)
        else:
            x_enc_prev = x

        for idx, layer in enumerate(self.encoder):
            x_enc_prev = layer(x_enc_prev)
            if not isinstance(layer, nn.Linear):
                x_encs.append(x_enc_prev)
            else:
                x_encs.append(None)

        for idx, x_enc in enumerate(x_encs):
            if x_enc is not None:
                x_rec_prev = x_enc
                for layer in self.decoder[-idx - 1 :]:
                    x_rec_prev = layer(x_rec_prev)
                x_recs.append(x_rec_prev)
        return torch.stack(x_recs, dim=0)

    def weight_recs(self, x_recs):
        alpha = self.alpha.view(-1, *[1] * (x_recs.dim() - 1))
        return torch.clip(torch.sum(x_recs * alpha, dim=0), min=0, max=1)

    def forward(self, x):
        x_recs = self.compute_recs(x)
        return self.weight_recs(x_recs)

    def learn_one(self, x: dict) -> anomaly.AnomalyDetector:
        if self.is_initialized is False:
            self._init_net(x.shape[1])

        x = prep_input(x)

        x_recs = self.compute_recs(x)
        x_rec = self.weight_recs(x_recs)

        loss = self.loss_fn(x_rec, x)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = torch.stack([self.loss_fn(x_rec, x) for x_rec in x_recs])
        with torch.no_grad():
            self.alpha = self.alpha * torch.pow(self.beta, losses)
            self.alpha = torch.max(self.alpha, self.alpha_min)
            self.alpha = self.alpha / torch.sum(self.alpha)

        return self

    def _init_net(self, n_features):
        if self.build_fn is None:
            self.build_fn = get_fc_autoencoder

        self.encoder, self.decoder = self.build_fn(
            n_features=n_features, **self._filter_args(self.build_fn)
        )
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        self.encoder = [
            module
            for module in self.encoder.modules()
            if not isinstance(module, nn.Sequential)
        ]
        self.decoder = [
            module
            for module in self.decoder.modules()
            if not isinstance(module, nn.Sequential)
        ]


        if isinstance(self.encoder[0], nn.Dropout):
            self.dropout = self.encoder.pop(0)

        self.optimizer = self.configure_optimizers()

        n_encoding_layers = len(
            [layer for layer in self.encoder if isinstance(layer, nn.Linear)]
        )

        self.register_buffer("alpha", torch.ones(n_encoding_layers) / n_encoding_layers)
        self.register_buffer("beta", torch.ones(n_encoding_layers) * self.beta_scalar)
        self.register_buffer("alpha_min", torch.tensor(self.s / n_encoding_layers))

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            nn.ModuleList(self.encoder + self.decoder).parameters(),
            **self._filter_args(self.optimizer_fn),
        )
        return optimizer


class ScoreStandardizer():
    def __init__(self, momentum=0.99, with_std=True):
        self.with_std = with_std
        self.momentum = momentum
        self.mean = None
        self.var = 0
    
    def learn_one(self, x):
        if self.mean is None:
            self.mean = x
        else:
            last_diff = x - self.mean
            self.mean += (1 - self.momentum) * last_diff
            if self.with_std:
                self.var = self.momentum * (self.var + (1-self.momentum)*last_diff)

    def transform_one(self, x):
        x_centered = x - self.mean
        if self.with_std:
            x_standardized = np.divide(x_centered, self.var ** 0.5, where=self.var>0)
        else:
            x_standardized = x_centered
        return x_standardized

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)
