from typing import Callable, Type, Union

import pandas as pd
import torch
from river.base.typing import RegTarget

from deep_river.base import RollingDeepEstimator
from deep_river.regression import Regressor
from deep_river.utils.tensor_conversion import deque2rolling_tensor, float2tensor


class _TestLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1
        self.lstm = torch.nn.LSTM(
            input_size=2, hidden_size=self.hidden_size, num_layers=1
        )

    def forward(self, X, **kwargs):
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(X)
        hn = hn.view(-1, self.hidden_size)
        return hn


class RollingRegressor(RollingDeepEstimator, Regressor):
    """
    Wrapper that feeds a sliding window of the most recent examples to the
    wrapped PyTorch regression model.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a
        loss function provided by `torch.nn.functional` or one of the
        following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer
        Optimizer to be used for training the wrapped model. Can be an
        optimizer class provided by `torch.optim` or one of the following:
        "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    is_feature_incremental
        Whether the model should adapt to the appearance of
        previously features by adding units to the input
        layer of the network.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Number of recent examples to be fed to the wrapped model at each step.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        window_size: int = 10,
        append_predict: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            window_size=window_size,
            append_predict=append_predict,
            seed=seed,
            **kwargs,
        )

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
            "module": _TestLSTM(),
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
            "lr": 1e-3,
            "is_feature_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However,
        in some cases, some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return set()

    def learn_one(self, x: dict, y: RegTarget, **kwargs) -> "Regressor":
        """
        Performs one step of training with the sliding
        window of the most recent examples.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        RollingRegressor
            The regressor itself.
        """
        self._update_observed_features(x)
        self._adapt_input_dim(x)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])

        if len(self._x_window) == self.window_size:
            x_t = deque2rolling_tensor(self._x_window, device=self.device)
            y_t = float2tensor(y, device=self.device)
            self._learn(x_t, y_t)

        return self

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> "Regressor":
        self._update_observed_features(X)
        self._adapt_input_dim(X)
        self._x_window.extend(X.values.tolist())
        if len(self._x_window) == self.window_size:
            x_t = deque2rolling_tensor(self._x_window, device=self.device)
            y_t = torch.unsqueeze(torch.tensor(y, device=self.device), 1)
            self._learn(x_t, y_t)

        return self

    def predict_one(self, x: dict):
        """
        Predicts the target value for the current sliding
        window of most recent examples.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        RegTarget
            Predicted target value.
        """
        res = 0.0
        self._update_observed_features(x)
        self._adapt_input_dim(x)
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win

        self.module.eval()
        with torch.inference_mode():
            x_t = deque2rolling_tensor(x_win, device=self.device)
            res = self.module(x_t).numpy(force=True).item()

        return res

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        self._update_observed_features(X)
        self._adapt_input_dim(X)
        X = X[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(X.values.tolist())
        if self.append_predict:
            self._x_window = x_win
        self.module.eval()
        with torch.inference_mode():
            x_t = deque2rolling_tensor(x_win, device=self.device)
            res = self.module(x_t).detach().tolist()
        return res
