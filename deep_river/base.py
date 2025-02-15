import collections
import inspect
import warnings
from typing import Any, Callable, Deque, Dict, List, Type, Union, cast

import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base
from river.time_series.base import Forecaster
from torch.utils.hooks import RemovableHandle

from deep_river.utils import get_loss_fn, get_optim_fn
from deep_river.utils.hooks import ForwardOrderTracker, apply_hooks
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
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
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
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.module_cls = module
        self.module: torch.nn.Module = cast(torch.nn.Module, None)
        self.loss_func = get_loss_fn(loss_fn)
        self.loss_fn = loss_fn
        self.optimizer_func = get_optim_fn(optimizer_fn)
        self.optimizer_fn = optimizer_fn
        self.is_feature_incremental = is_feature_incremental
        self.is_class_incremental: bool = False
        self.observed_features: OrderedSet[str] = OrderedSet([])
        self.lr = lr
        self.device = device
        self.kwargs = kwargs
        self.seed = seed
        self.input_layer = cast(torch.nn.Module, None)
        self.input_expansion_instructions = cast(Dict, None)
        self.output_layer = cast(torch.nn.Module, None)
        self.output_expansion_instructions = cast(Dict, None)
        self.module_initialized = False
        torch.manual_seed(seed)

    def _filter_kwargs(self, fn: Callable, override=None, **kwargs) -> dict:
        """Filters `net_params` and returns those in `fn`'s arguments.

        Parameters
        ----------
        fn
            Arbitrary function
        override
            Dictionary, values to override `torch_params`

        Returns
        -------
        dict
            Dictionary containing variables in both `sk_params` and
            `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in kwargs.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def draw(self) -> Digraph:
        """Draws the wrapped model."""
        first_parameter = next(self.module.parameters())
        input_shape = first_parameter.size()
        y_pred = self.module(torch.rand(input_shape))
        return make_dot(y_pred.mean(), params=dict(self.module.named_parameters()))

    def initialize_module(self, x: dict | pd.DataFrame, **kwargs):
        """
        Parameters
        ----------
        module
          The instance or class or callable to be initialized, e.g.
          ``self.module``.
        kwargs : dict
          The keyword arguments to initialize the instance or class. Can be an
          empty dict.
        Returns
        -------
        instance
          The initialized component.
        """
        torch.manual_seed(self.seed)
        if isinstance(x, Dict):
            n_features = len(x)
        elif isinstance(x, pd.DataFrame):
            n_features = len(x.columns)

        if not isinstance(self.module_cls, torch.nn.Module):
            self.module = self.module_cls(
                n_features=n_features,
                **self._filter_kwargs(self.module_cls, kwargs),
            )

        self.module.to(self.device)
        self.optimizer = self.optimizer_func(self.module.parameters(), lr=self.lr)
        self.module_initialized = True

        self._get_input_output_layers(n_features=n_features)

    def clone(
        self,
        new_params: dict[Any, Any] | None = None,
        include_attributes=False,
    ):
        """Clones the estimator.

        Parameters
        ----------
        new_params
            New parameters to be passed to the cloned estimator.
        include_attributes
            If True, the attributes of the estimator will be copied to the
            cloned estimator. This is useful when the estimator is a
            transformer and the attributes are the learned parameters.

        Returns
        -------
        DeepEstimator
            The cloned estimator.
        """
        new_params = new_params or {}
        new_params.update(self.kwargs)
        new_params.update(self._get_params())
        new_params.update({"module": self.module_cls})

        clone = self.__class__(**new_params)
        if include_attributes:
            clone.__dict__.update(self.__dict__)
        return clone

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

    def _get_input_output_layers(self, n_features: int):
        handles: List[RemovableHandle] = []
        tracker = ForwardOrderTracker()
        apply_hooks(module=self.module, hook=tracker, handles=handles)

        x_dummy = torch.empty((1, n_features), device=self.device)
        self.module(x_dummy)

        for h in handles:
            h.remove()

        if self.is_class_incremental:
            if tracker.ordered_modules and isinstance(
                tracker.ordered_modules[-1], SUPPORTED_LAYERS
            ):

                self.output_layer = tracker.ordered_modules[-1]
                self.output_expansion_instructions = load_instructions(
                    self.output_layer
                )
            else:
                warnings.warn(
                    "The model will not be able to adapt its output to new "
                    "classes since no supported output layer was found."
                )
                self.is_class_incremental = False

        if self.is_feature_incremental:
            if tracker.ordered_modules and isinstance(
                tracker.ordered_modules[0], SUPPORTED_LAYERS
            ):
                self.input_layer = tracker.ordered_modules[0]
                self.input_expansion_instructions = load_instructions(self.input_layer)
            else:
                warnings.warn(
                    "The model will not be able to adapt its input layer to "
                    "new features since no supported input layer was found."
                )
                self.is_feature_incremental = False


class RollingDeepEstimator(DeepEstimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers including a rolling window to allow the
    model to make predictions based on multiple previous examples.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped. The Module
        should accept parameter `n_features` so that the returned model's
        input shape can be determined based on the number of features in the
        initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
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
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
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
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )

        self.window_size = window_size
        self.append_predict = append_predict
        self._x_window: Deque = collections.deque(maxlen=window_size)
        self._batch_i = 0


from collections import OrderedDict
from typing import Callable, Dict, List, Type, Union

import pandas as pd
import torch
from river import base


class DeepForecaster(RollingDeepEstimator, Forecaster):
    """
    A deep learning-based forecaster using a rolling window for sequential data.
    Combines the functionality of River's Forecaster and the RollingDeepEstimator.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
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
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
            **kwargs,
        )

    def learn_one(self, y: float, x: dict | None = None) -> None:
        """
        Updates the model using one supervised learning step.

        Parameters
        ----------
        y : float
            The target value.
        x : dict | None
            Additional features.
        """
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)
        # Update rolling window
        x = x or {}
        self._x_window.append(list(x.values()))

        # Prepare input and target tensors
        x_tensor = torch.tensor(
            list(self._x_window), dtype=torch.float32, device=self.device
        )
        y_tensor = torch.tensor([y], dtype=torch.float32, device=self.device)

        # Forward pass
        self.module.train()
        y_pred = self.module(x_tensor).squeeze()

        # Compute loss and backpropagation
        loss = self.loss_func(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forecast(self, horizon: int, xs: list[dict] | None = None) -> list:
        """
        Generates forecasts for the specified horizon.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        xs : list[dict] | None
            Optional features for each step in the horizon.

        Returns
        -------
        list
            Forecasted values.
        """
        if not self.module_initialized:
            raise ValueError(
                "you need to call the learn_one function at least once before calling forecast."
            )
        self.module.eval()
        forecasts = []
        rolling_window = collections.deque(self._x_window, maxlen=self.window_size)

        for t in range(horizon):
            x = xs[t] if xs else {}
            x_tensor = torch.tensor(
                list(rolling_window), dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                y_pred = self.module(x_tensor).squeeze().item()
                forecasts.append(y_pred)

            if self.append_predict:
                # Update the rolling window with the new prediction
                rolling_window.append(list(x.values()) if x else [y_pred])

        return forecasts
