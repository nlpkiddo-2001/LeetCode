# 1. Finetune a pre trained model
from math import sqrt

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

dataset = load_dataset("glue", "stt2")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

acc_metric = evaluate.load("accuracy")
def compute_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions, labels)

training_args = TrainingArguments(
    output_dir="dummy",
    per_device_train_batch_size=2,
    num_train_epochs=1
)
trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metric
)
trainer.train()
#####################################################################################################################################################################################################################
# 2. K - nearest neighbour implementation from scratch
from math import sqrt


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

prediction = predict_classification(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
#####################################################################################################################################################################################################################
# 3. End - to - End Pipeline for a ML Model given a dataset

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_depth" : [50, 100, 200],
    "max_depth" : [3, 5, 7],
    "min_sample_split" : [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


####################################################################################################################################################################################################################
# 4. Conv2d from scratch

import numpy as np

def conv2d(image, kernel, stride=1):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            region = image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [3, 1, 0, 1, 2],
    [2, 3, 2, 0, 1],
    [1, 0, 1, 3, 2]
])

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

print(conv2d(image, kernel))
####################################################################################################################################################################################################################
# 5. Self attention

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(X):
    d_k = X.shape[-1]

    W_q = np.random.randn(d_k, d_k)
    W_k = np.random.randn(d_k, d_k)
    W_v = np.random.randn(d_k, d_k)

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = Q @ K.T / np.sqrt(d_k)

    weights = softmax(scores)

    output = weights @ V

    return output

X = np.array([
    [1, 0, 1, 0],  # "The"
    [0, 2, 0, 2],  # "cat"
    [1, 1, 1, 1]   # "sat"
], dtype=float)

output, attention_weights = self_attention(X)

print("Attention Weights:\n", np.round(attention_weights, 3))
print("\nOutput representations:\n", np.round(output, 3))
####################################################################################################################################################################################################################
# 6. Batch normalization

import numpy as np

class BatchNorm:
    def __init__(self, epsilon = 0.5, momentum = 0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 1
        self.gamma = 1
        self.beta = 0

    def forward(self, X, training=True):
        if training:
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)

            out = self.gamma * X_norm + self.beta

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )

            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_mean
            )

        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_norm + self.beta

        return out
####################################################################################################################################################################################################################
# 7. Encoder Block
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, attn_mask=None):
        attn_out = self.self_attn(
            x,x,x,
            attn_mask=attn_mask
        )

        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ffn(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x

B, T, C = 2, 5, 32
x = torch.randn(B, T, C)
block = EncoderBlock(d_model=C, num_heads=4)
out = block(x)
print(out.shape)
####################################################################################################################################################################################################################
