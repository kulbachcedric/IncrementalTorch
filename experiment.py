import math
import calendar

from river import datasets, metrics, compose
import torch
import torch.nn as nn
import datetime as dt

from deep_river.base import DeepForecaster


# Define a simple PyTorch module for forecasting
class SimpleForecasterModule(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden = nn.Linear(n_features, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

def get_month_distances(x):
    return {
        calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
        for month in range(1, 13)
    }

def get_ordinal_date(x):
    return {'ordinal_date': x['month'].toordinal()}

extract_features = compose.TransformerUnion(
    get_ordinal_date,
    get_month_distances
)

model = (
    extract_features |
    DeepForecaster(
    module=SimpleForecasterModule,
    loss_fn="mse",
    optimizer_fn="adam",
    lr=0.01,
    device="cpu",
    ))


horizon = 12
future = [
    {'month': dt.date(year=1961, month=m, day=1)}
    for m in range(1, horizon + 1)
]

for x, y in datasets.AirlinePassengers():
    model.learn_one(x, y)

forecast = model.forecast(horizon=horizon)
for x, y_pred in zip(future, forecast):
    print(x['month'], f'{y_pred:.3f}')

