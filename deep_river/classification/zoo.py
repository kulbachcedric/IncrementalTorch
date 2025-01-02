from typing import Callable, Union

from torch import nn

from deep_river.classification import Classifier


class LogisticRegression(Classifier):
    """
    This class implements a logistic regression model in PyTorch.

    Parameters
    ----------
    module
        An initialized PyTorch model for logistic regression. If not provided,
        a default logistic regression module is created internally.
    loss_fn
        Loss function to be used for training the model. Can be a loss function
        provided by `torch.nn.functional` or one of the following: 'mse', 'l1',
        'cross_entropy', 'binary_cross_entropy_with_logits',
        'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer
        Optimizer to be used for training the model. Can be an optimizer class
        provided by `torch.optim` or one of the following: "adam", "adam_w",
        "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    output_is_logit
        Whether the module produces logits as output. If true, either softmax
        or sigmoid is applied to the outputs when predicting.
    is_class_incremental
        Whether the classifier should adapt to the appearance of previously
        unobserved classes by adding a unit to the output layer of the network.
    is_feature_incremental
        Whether the model should adapt to the appearance of new features by
        adding units to the input layer of the network.
    device
        Device to run the model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the model.
    **kwargs
        Additional parameters to pass to the logistic regression module if the
        default module is used.

    Examples
    --------
    >>> from deep_river.classification import LogisticRegression
    >>> from river import metrics, preprocessing, compose, datasets
    >>> from torch import manual_seed

    >>> _ = manual_seed(42)

    >>> model_pipeline = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     LogisticRegression()
    ... )

    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> for x, y in dataset:
    ...     y_pred = model_pipeline.predict_one(x) # make a prediction
    ...     metric.update(y, y_pred) # update the metric
    ...     model_pipeline.learn_one(x, y) # update the model

    >>> print(f"Accuracy: {metric.get():.2f}")
    Accuracy: 0.56
    """

    class DefaultLRModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense0 = nn.Linear(2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, **kwargs):
            x = self.dense0(x)
            return self.sigmoid(x)

    def __init__(
        self,
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
        self.module = LogisticRegression.DefaultLRModule()
        super().__init__(
            module=self.module,
            loss_fn=loss_fn,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            is_feature_incremental=is_feature_incremental,
            optimizer=optimizer,
            device=device,
            lr=lr,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit testing the
        respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the class.
        """
        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_class_incremental": True,
        }


class MultiLayerPerceptron(Classifier):
    """
    This class implements a logistic regression model in PyTorch.

    Parameters
    ----------
    n_width
        Number of units in each hidden layer.
    n_layers
        Number of hidden layers.
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
        previously unobserved classes by adding an unit to the output
        layer of the network. This works only if the last trainable
        layer is an nn.Linear layer. Note also, that output activation
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
    **kwargs
        Parameters to be passed to the `build_fn` function aside from
        `n_features`.

    Examples
    --------
    >>> from deep_river.classification import MultiLayerPerceptron
    >>> from river import metrics, preprocessing, compose, datasets
    >>> from torch import nn, manual_seed

    >>> _ = manual_seed(42)

    >>> model_pipeline = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     MultiLayerPerceptron(n_width=5,n_layers=5)
    ... )

    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> for x, y in dataset:
    ...     y_pred = model_pipeline.predict_one(x) # make a prediction
    ...     metric.update(y, y_pred) # update the metric
    ...     model_pipeline.learn_one(x, y) # update the model

    >>> print(f"Accuracy: {metric.get():.2f}")
    Accuracy: 0.44

    """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_layers):
            super().__init__()
            hidden = [nn.Linear(2, n_width)]
            hidden += [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, 1)

        def forward(self, x, **kwargs):
            for layer in self.hidden:
                x = layer(x)
                x = nn.functional.sigmoid(x)
            return self.denselast(x)

    def __init__(
        self,
        n_width: int = 5,
        n_layers: int = 5,
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

        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=MultiLayerPerceptron.MLPModule(n_width, n_layers),
            loss_fn=loss_fn,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            is_feature_incremental=is_feature_incremental,
            optimizer=optimizer,
            device=device,
            lr=lr,
            seed=seed,
            n_width=n_width,
            n_layers=n_layers,
            **kwargs,
        )
        self.n_width = n_width
        self.n_layers = n_layers

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
            Set of checks to skip during unit testing.
        """
        return {"check_predict_proba_one"}
