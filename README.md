<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>


![PyPI](https://img.shields.io/pypi/v/river_torch)
[![unit-tests](https://github.com/online-ml/river-torch/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/kulbachcedric/DeepRiver/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/online-ml/river-torch/branch/master/graph/badge.svg?token=ZKUIISZAYA)](https://codecov.io/gh/online-ml/river-torch)
[![docs](https://github.com/online-ml/river-torch/actions/workflows/mkdocs.yml/badge.svg)](https://github.com/online-ml/river-torch/actions/workflows/unit_test.yml)


<p align="center">
    river-torch is a Python library for online deep learning.
    River-torch's ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks.
    It combines the <a href="https://www.riverml.xyz">river</a> API with the capabilities of designing neural networks based on <a href="https://pytorch.org">PyTorch</a>.
</p>

## 💈 Installation

```shell
pip install river-torch
```

You can install the latest development version from GitHub as so:

```shell
pip install https://github.com/online-ml/river-torch/archive/refs/heads/master.zip
```

## 🍫 Quickstart

We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="https://online-ml.github.io/river-torch">Documentation</a>.

### Classification

```python
>>> from river.datasets import Phishing
>>> from river import metrics
>>> from river import preprocessing
>>> from river import compose
>>> from river_torch import classification
>>> from torch import nn
>>> from torch import optim
>>> from torch import manual_seed

>>> _ = manual_seed(42)

>>> class MyModule(nn.Module):
...     def __init__(self, n_features):
...         super(MyModule, self).__init__()
...         self.dense0 = nn.Linear(n_features, 5)
...         self.nonlin = nn.ReLU()
...         self.dense1 = nn.Linear(5, 2)
...         self.softmax = nn.Softmax(dim=-1)
...
...     def forward(self, X, **kwargs):
...         X = self.nonlin(self.dense0(X))
...         X = self.nonlin(self.dense1(X))
...         X = self.softmax(X)
...         return X

>>> model = compose.Pipeline(
... preprocessing.StandardScaler(),
... classification.Classifier(module=MyModule, loss_fn='binary_cross_entropy', optimizer_fn='sgd')
... )

>>> dataset = Phishing()
>>> metric = metrics.Accuracy()

>>> for x, y in dataset:
...     y_pred = model.predict_one(x)  # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model = model.learn_one(x)  # make the model learn

```

### Anomaly Detection

```python
>>> from river_torch.anomaly import Autoencoder
>>> from river import metrics
>>> from river.datasets import CreditCard
>>> from torch import nn
>>> import math
>>> from river.compose import Pipeline
>>> from river.preprocessing import MinMaxScaler

>>> dataset = CreditCard().take(5000)
>>> metric = metrics.ROCAUC(n_thresholds=50)

>>> class MyAutoEncoder(nn.Module):
...     def __init__(self, n_features, latent_dim=3):
...         super(MyAutoEncoder, self).__init__()
...         self.linear1 = nn.Linear(n_features, latent_dim)
...         self.nonlin = nn.LeakyReLU()
...         self.linear2 = nn.Linear(latent_dim, n_features)
...
...     def forward(self, X, **kwargs):
...         X = self.linear1(X)
...         X = self.nonlin(X)
...         X = self.linear2(X)
...         return nn.functional.sigmoid(X)

>>> ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
>>> scaler = MinMaxScaler()
>>> model = Pipeline(scaler, ae)

>>> for x, y in dataset:
...    score = model.score_one(x)
...    model = model.learn_one(x=x)
...    metric = metric.update(y, score)
...
>>> print(f"ROCAUC: {metric.get():.4f}")
ROCAUC: 0.7447

```

## 🏫 Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>
