from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base
from river.base.typing import ClfTarget

from deep_river.base import DeepEstimator
from deep_river.utils.layer_adaptation import expand_layer
from deep_river.utils.tensor_conversion import (
    df2tensor,
    dict2tensor,
    labels2onehot,
    output2proba,
)


class _TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = torch.nn.Linear(2, 5)
        self.nonlinear = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.nonlinear(self.dense0(x))
        x = self.nonlinear(self.dense1(x))
        return x


class Classifier(DeepEstimator, base.MiniBatchClassifier):
    """
    Wrapper for PyTorch classification models that automatically handles
    increases in the number of classes by adding output neurons in case
    the number of observed classes exceeds the current
    number of output neurons.

    Parameters
    ----------
    module
        An initialized PyTorch model to be wrapped.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a
        loss function provided by `torch.nn.functional` or one of the
        following: 'mse', 'l1', 'cross_entropy',
        'binary_cross_entropy_with_logits', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    output_is_logit
        Whether the module produces logits as output. If true, either
        softmax or sigmoid is applied to the outputs when predicting.
    is_class_incremental
        Whether the classifier should adapt to the appearance of
        previously unobserved classes by adding a unit to the output
        layer of the network. This works only if the last trainable
        layer is a nn.Linear layer. Note also, that output activation
        functions can not be adapted, meaning that a binary classifier
        with a sigmoid output can not be altered to perform multi-class
        predictions.
    is_feature_incremental
        Whether the model should adapt to the appearance of
        previously features by adding units to the input
        layer of the network.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.

    Examples
    --------
    >>> from river import metrics, preprocessing, compose, datasets
    >>> from deep_river import classification
    >>> from torch import nn, manual_seed

    >>> manual_seed(42)

    >>> class MyModule(nn.Module):
    ...     def __init__(self, n_features):
    ...         super(MyModule, self).__init__()
    ...         self.dense0 = nn.Linear(n_features, 5)
    ...         self.nlin = nn.ReLU()
    ...         self.dense1 = nn.Linear(5, 2)
    ...         self.softmax = nn.Softmax(dim=-1)
    ...
    ...     def forward(self, x, **kwargs):
    ...         x = self.nlin(self.dense0(x))
    ...         x = self.nlin(self.dense1(x))
    ...         x = self.softmax(x)
    ...         return x

    >>> model = MyModule(n_features=2)
    >>> classifier = Classifier(
    ...     module=model,
    ...     loss_fn="binary_cross_entropy",
    ...     optimizer="adam",
    ...     is_class_incremental=True,
    ...     is_feature_incremental=True,
    ... )

    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> for x, y in dataset:
    ...     y_pred = classifier.predict_one(x)
    ...     metric.update(y, y_pred)
    ...     classifier.learn_one(x, y)

    >>> print(f"Accuracy: {metric.get()}")
    Accuracy: 0.7264
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "The 'module' parameter must be an initialized PyTorch model."
            )

        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )
        self.observed_classes: OrderedSet[ClfTarget] = OrderedSet([])
        self.output_is_logit = output_is_logit
        self.is_class_incremental = is_class_incremental

    @classmethod
    def _unit_test_params(cls):
        yield {
            "module": _TestModule(),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_class_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite.
        However, in some cases, some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            of checks to skip during unit testing.
        """
        return set()

    def _learn(self, x: torch.Tensor, y: Union[ClfTarget, pd.Series]):
        self.module.train()
        self.optimizer.zero_grad()
        y_pred = self.module(x)
        n_classes = y_pred.shape[-1]
        y_onehot = labels2onehot(
            y=y,
            classes=self.observed_classes,
            n_classes=n_classes,
            device=self.device,
        )
        loss = self.loss_fn(y_pred, y_onehot)
        loss.backward()
        self.optimizer.step()
        return self

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> "Classifier":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        Classifier
            The classifier itself.
        """

        self._update_observed_features(x)
        self._update_observed_classes(y)

        # check last layer
        self._adapt_input_dim(x)
        self._adapt_output_dim(y)

        x_t = dict2tensor(x, features=self.observed_features, device=self.device)

        return self._learn(x=x_t, y=y)

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """
        Predict the probability of each label given the input.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        Dict[ClfTarget, float]
            Dictionary of probabilities for each label.
        """

        self._update_observed_features(x)
        self._adapt_input_dim(x)

        x_t = dict2tensor(x, features=self.observed_features, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)
        return output2proba(y_pred, self.observed_classes, self.output_is_logit)[0]

    def _update_observed_classes(self, y) -> bool:
        n_existing_classes = len(self.observed_classes)
        if isinstance(y, Union[ClfTarget, np.bool_]):  # type: ignore[arg-type]
            self.observed_classes.add(y)
        else:
            self.observed_classes |= y

        if len(self.observed_classes) > n_existing_classes:
            self.observed_classes = OrderedSet(sorted(self.observed_classes))
            return True
        else:
            return False

    def _adapt_output_dim(self, y: ClfTarget | pd.Series):
        has_new_class = self._update_observed_classes(y)
        if (
            has_new_class
            and len(self.observed_classes) > 2
            and self.is_class_incremental
        ):
            expand_layer(
                self.output_layer,
                self.output_expansion_instructions,
                len(self.observed_classes),
                output=True,
            )

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> "Classifier":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X
            Input examples.
        y
            Target values.

        Returns
        -------
        Classifier
            The classifier itself.
        """
        # check if model is initialized

        self._update_observed_features(X)
        self._update_observed_classes(y)

        self._adapt_input_dim(X)
        self._adapt_output_dim(y)

        x_t = df2tensor(X, features=self.observed_features, device=self.device)

        return self._learn(x=x_t, y=y)

    def predict_proba_many(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probability of each label given the input.

        Parameters
        ----------
        x
            Input examples.

        Returns
        -------
        pd.DataFrame
            of probabilities for each label.
        """
        self._update_observed_features(x)

        self._adapt_input_dim(x)
        x_t = df2tensor(x, features=self.observed_features, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(x_t)
        return pd.DataFrame(
            output2proba(y_preds, self.observed_classes, self.output_is_logit)
        )
