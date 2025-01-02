from typing import Any, Callable, List, Type, Union

import numpy as np
import pandas as pd
import torch
from river import anomaly
from torch import nn

from deep_river.base import RollingDeepEstimator
from deep_river.utils.tensor_conversion import deque2rolling_tensor


class _TestLSTMAutoencoder(nn.Module):
    def __init__(self, hidden_size=30, n_layers=1, batch_first=False):
        super().__init__()
        self.n_features = 2
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.time_axis = 1 if batch_first else 0
        self.encoder = nn.LSTM(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        output, (h, c) = self.encoder(x)
        return output


class RollingAutoencoder(RollingDeepEstimator, anomaly.base.AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the network's reconstruction
    error for scoring the anomalousness of a given example. It supports a rolling
    window for making predictions based on the reconstructability of multiple
    previous examples.

    Parameters
    ----------
    module : torch.nn.Module
        An instance of a PyTorch module that defines the autoencoder.
        The module should accept inputs with shape `(window_size, batch_size, n_features)`
        and adapt to the initial training example during initialization.
    loss_fn : Union[str, Callable], default="mse"
        Loss function for training. Can be a function from `torch.nn.functional` or
        one of: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer : Union[str, Callable], default="sgd"
        Optimizer for training. Can be a class from `torch.optim` or one of:
        "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr : float, default=1e-3
        Learning rate for the optimizer.
    device : str, default="cpu"
        Device to run the model on, either "cpu" or "cuda".
    seed : int, default=42
        Random seed for reproducibility.
    window_size : int, default=10
        Size of the rolling window for storing previous examples.
    append_predict : bool, default=False
        If True, appends inputs passed during prediction to the rolling window.
    **kwargs
        Additional parameters for the optimizer or other configurations.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> from deep_river.anomaly import RollingAutoencoder

    >>> class SimpleAutoencoder(nn.Module):
    >>>     def __init__(self, n_features):
    >>>         super().__init__()
    >>>         self.encoder = nn.Linear(n_features, 4)
    >>>         self.decoder = nn.Linear(4, n_features)

    >>>     def forward(self, x):
    >>>         x = self.encoder(x)
    >>>         x = self.decoder(x)
    >>>         return x

    >>> autoencoder = SimpleAutoencoder(n_features=5)
    >>> model = RollingAutoencoder(
    >>>     module=autoencoder,
    >>>     loss_fn="mse",
    >>>     optimizer="adam",
    >>>     lr=0.001,
    >>>     window_size=3,
    >>> )
    >>> X = pd.DataFrame([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    >>> model.learn_many(X)
    >>> scores = model.score_many(X)
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr=lr,
            device=device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        """
        Parameters for unit testing.

        Returns
        -------
        dict
            Dictionary of parameters for unit tests.
        """
        yield {
            "module": _TestLSTMAutoencoder(),
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Returns checks to skip during unit testing.

        Returns
        -------
        set
            Set of skipped test checks.
        """
        return {
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def _learn(self, x: torch.Tensor):
        """
        Trains the model on a batch of input data.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for training.
        """
        self.module.train()
        x_pred = self.module(x)
        loss = self.loss_fn(x_pred, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x: dict, y: Any = None, **kwargs) -> "RollingAutoencoder":
        """
        Trains the model with a single example.

        Parameters
        ----------
        x : dict
            Single input example.

        Returns
        -------
        RollingAutoencoder
            The estimator instance.
        """
        self._update_observed_features(x)
        self._x_window.append(list(x.values()))

        if len(self._x_window) == self.window_size:
            x_t = deque2rolling_tensor(self._x_window, device=self.device)
            self._learn(x_t)
        return self

    def learn_many(self, X: pd.DataFrame, y=None) -> "RollingAutoencoder":
        """
        Trains the model with a batch of examples.

        Parameters
        ----------
        X : pd.DataFrame
            Batch of input examples.

        Returns
        -------
        RollingAutoencoder
            The estimator instance.
        """
        self._update_observed_features(X)
        self._x_window.append(X.values.tolist())

        if len(self._x_window) == self.window_size:
            X_t = deque2rolling_tensor(self._x_window, device=self.device)
            self._learn(X_t)
        return self

    def score_one(self, x: dict) -> float:
        """
        Computes an anomaly score for a single example.

        Parameters
        ----------
        x : dict
            Single input example.

        Returns
        -------
        float
            Anomaly score.
        """
        self._update_observed_features(x)

        if len(self._x_window) == self.window_size:
            return self._compute_score(x)
        if self.append_predict:
            self._x_window.append(list(x.values()))
        return 0.0

    def score_many(self, X: pd.DataFrame) -> List[Any]:
        """
        Computes anomaly scores for a batch of examples.

        Parameters
        ----------
        X : pd.DataFrame
            Batch of input examples.

        Returns
        -------
        List[float]
            List of anomaly scores.
        """
        self._update_observed_features(X)

        x_win = self._x_window.copy()
        x_win.append(X.values.tolist())

        if len(self._x_window) == self.window_size:
            return self._compute_batch_score(x_win)
        if self.append_predict:
            self._x_window.append(X.values.tolist())
        return [0.0] * len(X)

    def _compute_score(self, x):
        """
        Computes the anomaly score for a single example.

        Parameters
        ----------
        x : dict
            Single input example.

        Returns
        -------
        float
            Anomaly score.
        """
        x_win = self._x_window.copy()
        x_win.append(list(x.values()))
        x_t = deque2rolling_tensor(x_win, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            x_pred = self.module(x_t)
        loss = self.loss_fn(x_pred, x_t)
        return loss.item()

    def _compute_batch_score(self, x_win):
        """
        Computes the anomaly scores for a batch of examples.

        Parameters
        ----------
        x_win : list
            List of input data in rolling window format.

        Returns
        -------
        List[float]
            List of anomaly scores for the batch.
        """
        X_t = deque2rolling_tensor(x_win, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            x_pred = self.module(X_t)
        loss = torch.mean(
            self.loss_fn(x_pred, X_t, reduction="none"),
            dim=list(range(1, X_t.dim())),
        )
        return loss.detach().cpu().numpy().tolist()
