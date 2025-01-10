import torch
import torch.nn as nn
from river import datasets, compose
from river.linear_model import LogisticRegression
from river.metrics import Accuracy
from river.preprocessing import StandardScaler

from deep_river.preprocessing import EmbeddingTransformer


# Define the model
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=50):
        super(WordEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)  # Example output for binary classification

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded.mean(dim=1)  # Mean pooling

# Initialize the embedding transformer
embedding_model = WordEmbeddingModel
transformer = EmbeddingTransformer(
    module=embedding_model,
    loss_fn="mse_loss",
    optimizer_fn="adam",
    tokenizer="basic_english",
    lr=0.01
)

# Combine the transformer with a logistic regression classifier
model = compose.Pipeline(
    ("embedding", transformer),
    ("scale", StandardScaler()),  # Scale the embedding vectors
    ("learn", LogisticRegression())
)

# Load the SMS spam dataset
dataset = datasets.SMSSpam()

# Define a metric
metric = Accuracy()

# Train the model
for i, (x, y) in enumerate(dataset):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)

# Example usage
test_instance = {"message": "Win a $1000 gift card now!"}
prediction = model.predict_one(test_instance)
print(f"Prediction for '{test_instance['message']}': {prediction}")
