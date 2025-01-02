import math
from typing import Any, Callable, Type, Union

import pandas as pd
import torch
from river import stats, utils
from scipy.special import ndtr

from deep_river.anomaly import ae
from deep_river.utils import dict2tensor
from deep_river.utils.tensor_conversion import df2tensor


class ProbabilityWeightedAutoencoder(ae.Autoencoder):
    """
    Wrapper for PyTorch autoencoder models for anomaly detection. Adjusts the learning rate based on the outlier probability estimate
    of input examples. If the probability exceeds the `skip_threshold`, the learning rate is reduced accordingly.

    Parameters
    ----------
    module : torch.nn.Module
        Fully initialized PyTorch autoencoder model.
    loss_fn : Union[str, Callable]
        Loss function for training (e.g., 'mse', 'l1') or a callable.
    optimizer : Union[str, Callable]
        Optimizer for training (e.g., 'adam', 'sgd') or a callable.
    lr : float
        Base learning rate.
    skip_threshold : float
        Probability threshold for adjusting the learning rate.
    device : str
        Device to run the model on ('cpu' or 'cuda').
    seed : int
        Random seed for reproducibility.
    window_size : int
        Window size for rolling statistics used in outlier probability estimation.
    **kwargs
        Additional parameters for customization.

    Examples
    --------
    >>> from deep_river.anomaly import ProbabilityWeightedAutoencoder
    >>> from river import metrics
    >>> from river.datasets import CreditCard
    >>> from torch import nn, manual_seed
    >>> import math
    >>> from river.compose import Pipeline
    >>> from river.preprocessing import MinMaxScaler

    >>> _ = manual_seed(42)
    >>> dataset = CreditCard().take(5000)
    >>> metric = metrics.RollingROCAUC(window_size=5000)

    >>> class MyAutoEncoder(torch.nn.Module):
    ...     def __init__(self, n_features, latent_dim=3):
    ...         super(MyAutoEncoder, self).__init__()
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

    >>> ae = ProbabilityWeightedAutoencoder(module=MyAutoEncoder, lr=0.005)
    >>> scaler = MinMaxScaler()
    >>> model = Pipeline(scaler, ae)

    >>> for x, y in dataset:
    ...    score = model.score_one(x)
    ...    model.learn_one(x=x)
    ...    metric.update(y, score)
    ...
    >>> print(f"Rolling ROCAUC: {metric.get():.4f}")
    Rolling ROCAUC: 0.8530
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        skip_threshold: float = 0.9,
        window_size: int = 250,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )
        self.skip_threshold = skip_threshold
        self.window_size = window_size
        self.rolling_mean = utils.Rolling(stats.Mean(), window_size=window_size)
        self.rolling_var = utils.Rolling(stats.Var(), window_size=window_size)

    def learn_one(
        self, x: dict, y: Any = None, **kwargs
    ) -> "ProbabilityWeightedAutoencoder":
        """
        Trains the model with a single example, adjusting the learning rate based on the outlier probability.

        Parameters
        ----------
        x : dict
            Input example.

        Returns
        -------
        ProbabilityWeightedAutoencoder
            The updated model.
        """
        self._update_observed_features(x)

        self._adapt_input_dim(x)
        x_t = dict2tensor(x, self.observed_features, device=self.device)

        self.module.train()
        x_pred = self.module(x_t)
        loss = self.loss_fn(x_pred, x_t)
        self._apply_adjusted_loss(loss)
        return self

    def learn_many(self, X: pd.DataFrame) -> "ProbabilityWeightedAutoencoder":
        """
        Trains the model with a batch of examples, adjusting the learning rate based on outlier probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Batch of input examples.

        Returns
        -------
        ProbabilityWeightedAutoencoder
            The updated model.
        """
        self._update_observed_features(X)

        self._adapt_input_dim(X)
        X_t = df2tensor(X, features=self.observed_features, device=self.device)

        self.module.train()
        x_pred = self.module(X_t)
        loss = torch.mean(
            self.loss_fn(x_pred, X_t, reduction="none"),
            dim=list(range(1, X_t.dim())),
        )
        self._apply_adjusted_loss(loss)
        return self

    def _apply_adjusted_loss(self, loss: torch.Tensor):
        """
        Scales the loss based on the outlier probability and applies backpropagation.

        Parameters
        ----------
        loss : torch.Tensor
            Computed loss for the input data.
        """
        loss_np = loss.detach().cpu().numpy()

        # Update rolling mean and variance
        if loss_np.ndim == 0:  # Single loss value
            self.rolling_mean.update(loss_np)
            self.rolling_var.update(loss_np)
        else:  # Batch of loss values
            for loss_val in loss_np:
                self.rolling_mean.update(loss_val)
                self.rolling_var.update(loss_val)

        # Normalize loss and compute outlier probability
        mean = self.rolling_mean.get()
        var = self.rolling_var.get() or 1  # Avoid division by zero
        loss_scaled = (loss_np - mean) / math.sqrt(var)
        prob = ndtr(loss_scaled)  # Outlier probability based on normalized loss

        # Adjust loss based on probability
        adjustment = (self.skip_threshold - prob) / self.skip_threshold
        adjusted_loss = torch.tensor(adjustment, device=self.device) * loss

        # Backpropagation
        self.optimizer.zero_grad()
        adjusted_loss.backward()
        self.optimizer.step()
