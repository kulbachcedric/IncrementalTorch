import collections
from typing import Callable, Deque, Dict, Union, cast

import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base

from deep_river.utils import get_loss_fn, get_optim_fn
from deep_river.utils.layer_adaptation import (
    SUPPORTED_LAYERS,
    expand_layer,
    load_instructions,
)

try:
    from graphviz import Digraph
    from torchviz import make_dot
except ImportError as e:
    raise ValueError("You have to install graphviz to use the draw method") from e


class DeepEstimator(base.Estimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers.

    Parameters
    ----------
    module
        An already initialized PyTorch module to be wrapped.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Additional parameters to be passed to the optimizer.
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
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "The `module` parameter must be an initialized PyTorch module."
            )

        self.module = module.to(device)
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer = get_optim_fn(optimizer)(self.module.parameters(), lr=lr)
        self.lr = lr
        self.device = device
        self.seed = seed
        self.kwargs = kwargs
        self.is_feature_incremental = is_feature_incremental

        self.observed_features: OrderedSet[str] = OrderedSet([])
        self.input_layer = cast(torch.nn.Module, None)
        self.input_expansion_instructions = cast(Dict, None)

        torch.manual_seed(seed)

    def draw(self) -> Digraph:
        """Draws the wrapped model."""
        first_parameter = next(self.module.parameters())
        input_shape = first_parameter.size()
        y_pred = self.module(torch.rand(input_shape))
        return make_dot(y_pred.mean(), params=dict(self.module.named_parameters()))

    def _adapt_input_dim(self, x: Dict | pd.DataFrame):
        has_new_feature = self._update_observed_features(x)

        if has_new_feature and self.is_feature_incremental:
            expand_layer(
                self.input_layer,
                self.input_expansion_instructions,
                len(self.observed_features),
                output=False,
            )

    def _update_observed_features(self, x):
        n_existing_features = len(self.observed_features)
        if isinstance(x, Dict):
            self.observed_features |= x.keys()
        else:
            self.observed_features |= x.columns

        if len(self.observed_features) > n_existing_features:
            self.observed_features = OrderedSet(sorted(self.observed_features))
            return True
        else:
            return False


class RollingDeepEstimator(DeepEstimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers including a rolling window to allow the
    model to make predictions based on multiple previous examples.

    Parameters
    ----------
    module
        An already initialized PyTorch module to be wrapped.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Size of the rolling window used for storing previous examples.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Additional parameters to be passed to the optimizer.
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
            **kwargs,
        )

        self.window_size = window_size
        self.append_predict = append_predict
        self._x_window: Deque = collections.deque(maxlen=window_size)
        self._batch_i = 0
