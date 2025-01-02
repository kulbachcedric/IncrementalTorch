from typing import Any, Callable, Dict, Type, Union

import numpy as np
import pandas as pd
import torch
from river.anomaly.base import AnomalyDetector
from torch import nn

from deep_river.base import DeepEstimator
from deep_river.utils import dict2tensor
from deep_river.utils.layer_adaptation import expand_layer
from deep_river.utils.tensor_conversion import df2tensor


class _TestAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=3):
        super().__init__()
        n_features = 2
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.nonlin = torch.nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)

    def forward(self, X, **kwargs):
        X = self.linear1(X)
        X = self.nonlin(X)
        X = self.linear2(X)
        return nn.functional.sigmoid(X)


class Autoencoder(DeepEstimator, AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the network's
    reconstruction error for scoring the anomalousness of a given example.

    Parameters
    ----------
    module : torch.nn.Module
        Fully initialized PyTorch autoencoder model.
    loss_fn : Union[str, Callable]
        Loss function for training. Can be one of the predefined loss functions
        ('mse', 'l1', etc.) or a callable.
    optimizer : Union[str, Callable]
        Optimizer for training. Can be a predefined optimizer ('adam', 'sgd', etc.) or a callable.
    lr : float
        Learning rate for the optimizer.
    is_feature_incremental : bool
        Whether to adapt to new input features dynamically.
    device : str
        Device to run the model on ('cpu' or 'cuda').
    seed : int
        Random seed for reproducibility.
    **kwargs
        Parameters to be passed to the `torch.Module` class
        aside from `n_features`.

    Examples
    --------
    >>> from deep_river.anomaly import Autoencoder
    >>> from river import metrics
    >>> from river.datasets import CreditCard
    >>> from torch import nn
    >>> import math
    >>> from river.compose import Pipeline
    >>> from river.preprocessing import MinMaxScaler

    >>> dataset = CreditCard().take(5000)
    >>> metric = metrics.RollingROCAUC(window_size=5000)

    >>> class MyAutoEncoder(torch.nn.Module):
    ...     def __init__(self, latent_dim=3):
    ...         super(MyAutoEncoder, self).__init__()
    ...         n_features = 2
    ...         self.linear1 = nn.Linear(n_features, latent_dim)
    ...         self.nonlin = torch.nn.LeakyReLU()
    ...         self.linear2 = nn.Linear(latent_dim, n_features)
    ...         self.sigmoid = nn.Sigmoid()
    ...
    ...     def forward(self, X, **kwargs):
    ...         X = self.linear1(X)
    ...         X = self.nonlin(X)
    ...         X = self.linear2(X)
    ...         return self.sigmoid(X)

    >>> ae = Autoencoder(module=MyAutoEncoder(), lr=0.005)
    >>> scaler = MinMaxScaler()
    >>> model = Pipeline(scaler, ae)

    >>> for x, y in dataset:
    ...    score = model.score_one(x)
    ...    model.learn_one(x=x)
    ...    metric.update(y, score)
    ...
    >>> print(f"Rolling ROCAUC: {metric.get():.4f}")
    Rolling ROCAUC: 0.8901
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            **kwargs,
        )
        self.is_class_incremental = is_feature_incremental

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit testing
        the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing
            the respective class.
        """

        yield {
            "module": _TestAutoencoder(),
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return set()

    def learn_one(self, x: dict, y: Any = None, **kwargs) -> "Autoencoder":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x : dict
            Input example.

        **kwargs

        Returns
        -------
        Autoencoder
            The updated model.
        """
        self._update_observed_features(x)
        self._adapt_input_dim(x)
        return self._learn(
            dict2tensor(x, features=self.observed_features, device=self.device)
        )

    def _learn(self, x: torch.Tensor) -> "Autoencoder":
        self.module.train()
        x_pred = self.module(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def score_one(self, x: dict) -> float:
        """
        Computes the anomaly score for a single example based on reconstruction error.

        Parameters
        ----------
        x : dict
            Input example.

        Returns
        -------
        float
            Anomaly score, where higher values indicate more anomalous examples.
        """
        self._update_observed_features(x)

        self._adapt_input_dim(x)
        x_t = dict2tensor(x, features=self.observed_features, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            x_pred = self.module(x_t)
        loss = self.loss_fn(x_pred, x_t).item()
        return loss

    def learn_many(self, X: pd.DataFrame) -> "Autoencoder":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X : pd.DataFrame
            Input batch of examples.

        Returns
        -------
        Autoencoder
            The updated model.
        """
        self._update_observed_features(X)
        self._adapt_input_dim(X)
        X_t = df2tensor(X, features=self.observed_features, device=self.device)
        return self._learn(X_t)

    def score_many(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes anomaly scores for a batch of examples.

        Parameters
        ----------
        X : pd.DataFrame
            Input batch of examples.

        Returns
        -------
        np.ndarray
            Anomaly scores for the batch.
        """
        self._update_observed_features(X)

        self._adapt_input_dim(X)
        X_t = df2tensor(X, features=self.observed_features, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            X_pred = self.module(X_t)
        loss = torch.mean(
            self.loss_fn(X_pred, X_t, reduction="none"),
            dim=list(range(1, X_t.dim())),
        )
        return loss.cpu().detach().numpy()

    def _adapt_input_dim(self, x: Union[Dict, pd.DataFrame]):
        has_new_feature = self._update_observed_features(x)

        if has_new_feature and self.is_feature_incremental:
            expand_layer(
                self.input_layer,
                self.input_expansion_instructions,
                len(self.observed_features),
                output=False,
            )
            expand_layer(
                self.output_layer,
                self.output_expansion_instructions,
                len(self.observed_features),
                output=True,
            )
