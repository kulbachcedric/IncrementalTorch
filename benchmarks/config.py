from model_zoo.torch import (TorchLinearRegression, TorchLogisticRegression,
                             TorchLSTMClassifier, TorchLSTMRegressor,
                             TorchMLPClassifier, TorchMLPRegressor)
from river import (dummy, evaluate, linear_model, neural_net, optim,
                   preprocessing, stats)

from deep_river.classification import Classifier as TorchClassifier
from deep_river.classification import \
    RollingClassifier as TorchRollingClassifier
from deep_river.regression import Regressor as TorchRegressor
from deep_river.regression import RollingRegressor as TorchRollingRegressor

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    evaluate.BinaryClassificationTrack(),
    evaluate.MultiClassClassificationTrack(),
    evaluate.RegressionTrack(),
]

MODELS = {
    "Binary classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(
                optimizer=optim.SGD(LEARNING_RATE)
            )
        )
    },
    "Multiclass classification": {
        "Torch Logistic Regression": (
            preprocessing.StandardScaler()
            | TorchClassifier(
                module=TorchLogisticRegression,
                loss_fn="binary_cross_entropy",
                optimizer="sgd",
                is_class_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Torch MLP": (
            preprocessing.StandardScaler()
            | TorchClassifier(
                module=TorchMLPClassifier,
                loss_fn="binary_cross_entropy",
                optimizer="sgd",
                is_class_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Torch LSTM": (
            preprocessing.StandardScaler()
            | TorchRollingClassifier(
                module=TorchLSTMClassifier,
                loss_fn="binary_cross_entropy",
                optimizer="sgd",
                is_class_incremental=True,
                lr=LEARNING_RATE,
                window_size=20,
                append_predict=False,
                hidden_size=10,
            )
        ),
        "[baseline] Last Class": dummy.NoChangeClassifier(),
    },
    "Regression": {
        "Torch Linear Regression": (
            preprocessing.StandardScaler()
            | TorchRegressor(
                module=TorchLinearRegression,
                loss_fn="mse",
                optimizer="sgd",
                lr=LEARNING_RATE,
            )
        ),
        "Torch MLP": (
            preprocessing.StandardScaler()
            | TorchRegressor(
                module=TorchMLPRegressor,
                loss_fn="mse",
                optimizer="sgd",
                lr=LEARNING_RATE,
            )
        ),
        "River MLP": preprocessing.StandardScaler()
        | neural_net.MLPRegressor(
            hidden_dims=(5,),
            activations=(
                neural_net.activations.ReLU,
                neural_net.activations.ReLU,
                neural_net.activations.Identity,
            ),
            optimizer=optim.SGD(1e-3),
            seed=42,
        ),
        "Torch LSTM": (
            preprocessing.StandardScaler()
            | TorchRollingRegressor(
                module=TorchLSTMRegressor,
                loss_fn="mse",
                optimizer="sgd",
                lr=LEARNING_RATE,
                window_size=20,
                append_predict=False,
                hidden_size=10,
            )
        ),
        "[baseline] Mean predictor": dummy.StatisticRegressor(stats.Mean()),
    },
}
