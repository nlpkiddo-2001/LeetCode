"""
================================================================================
ML CODING PRACTICE — MLE INTERVIEW PREP
================================================================================

TABLE OF CONTENTS
-----------------
[1]  Finetune a pre-trained model (HuggingFace)     — commented reference
[2]  K-Nearest Neighbours from scratch
[3]  End-to-End sklearn Pipeline                    — commented reference
[4]  Conv2D from scratch                            — commented reference
[5]  Self-Attention from scratch                    — commented reference
[6]  Batch Normalization                            — commented reference
[7]  Encoder Block (PyTorch)                        — commented reference
[8]  Linear Regression from scratch
[9]  Logistic Regression from scratch
[10] CNN Training loop (PyTorch)                    — commented reference
[11] Conv3D from scratch
[12] Decision Tree Classifier from scratch
[13] Feedforward NN with Backprop from scratch (Linear + GeLU + FFN)
[14] Multi-Layer Feedforward NN with Training
[15] K-Means Clustering
[16] Perceptron
[17] Linear SVM

TOPICS YOU HAVEN'T COVERED (high-priority for MLE interviews)
-------------------------------------------------------------
See the block "MISSING TOPICS — STUDY LIST" at the bottom of this file.

EXPECTED INTERVIEW QUESTIONS FROM WHAT YOU'VE WRITTEN
-----------------------------------------------------
See the block "LIKELY FOLLOW-UP QUESTIONS" at the bottom of this file.
================================================================================
"""

from math import sqrt
import numpy as np
from networkx.algorithms import threshold


# =============================================================================
# [1] FINETUNE A PRE-TRAINED MODEL (HuggingFace)
# =============================================================================
# Intuition:
#   Instead of training a huge model from scratch (billions of params, weeks of
#   GPU time), we take one that's already learned general language patterns
#   ("distilbert-base-uncased") and only nudge its weights on OUR small labelled
#   dataset. The model's early layers already know grammar/syntax; we mostly
#   teach it the new task (e.g. sentiment = positive/negative).
#
#   The classification head on top is randomly initialized (num_labels=2), so
#   most of the learning happens there while the backbone gets fine adjustments.
#
# NOTE: Original code had `load_dataset("glue", "stt2")` — that's a typo, the
#       correct config is "sst2".
# -----------------------------------------------------------------------------
# from datasets import load_dataset
# import evaluate
# from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
#                           TrainingArguments, Trainer)
#
# dataset = load_dataset("glue", "sst2")
#
# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name,
#                                                            num_labels=2)
#
# def tokenize_function(examples):
#     # Turn raw text into integer token IDs the model understands.
#     # `truncation=True` cuts sequences longer than the model's max length.
#     return tokenizer(examples["sentence"], truncation=True)
#
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
#
# acc_metric = evaluate.load("accuracy")
# def compute_metric(eval_pred):
#     logits, labels = eval_pred
#     # argmax over the last dim gives us the predicted class per example.
#     predictions = np.argmax(logits, axis=-1)
#     return acc_metric.compute(predictions=predictions, references=labels)
#
# training_args = TrainingArguments(
#     output_dir="dummy",
#     per_device_train_batch_size=2,
#     num_train_epochs=1,
# )
# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     compute_metrics=compute_metric,
# )
# trainer.train()


# =============================================================================
# [2] K-NEAREST NEIGHBOURS (KNN) FROM SCRATCH
# =============================================================================

def euclidean_distance(row1, row2):
    """
    Compute the straight-line distance between two points.

    Formula:  d(a, b) = sqrt( sum_i (a_i - b_i)^2 )

    Intuition:
        Pythagoras generalized to N dimensions. If two points are close in
        every feature, their squared differences sum small → distance small.
        We stop at `len(row1) - 1` because the LAST column is the class
        label, not a feature.
    """
    distance = 0.0
    for i in range(len(row1) - 1):        # skip the label column
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    """
    Find the k closest training rows to `test_row`.

    Flow:
        1. Compute distance from test_row to every training row.
        2. Sort ascending by distance.
        3. Return the first k rows (the closest ones).

    Complexity: O(N * d) per query for N training points, d features.
    For large datasets, use KD-tree / Ball tree / FAISS instead.
    """
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])   # sort by distance, ascending
    return [distances[i][0] for i in range(num_neighbors)]


def predict_classification(train, test_row, num_neighbors):
    """
    Classify by MAJORITY VOTE among the k nearest neighbours.

    Intuition:
        "Tell me who your friends are and I'll tell you who you are."
        We look at the k closest labelled points and pick the most common
        class among them.

    For regression instead: return `sum(output_values) / k` (average).
    """
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]   # last col = label
    return max(set(output_values), key=output_values.count)


# dataset = [[2.7810836,  2.550537003,  0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019,  1.850220317, 0],
#            [3.06407232,  3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367,  1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]
#
# prediction = predict_classification(dataset, dataset[0], 3)
# print('Expected %d, Got %d.' % (dataset[0][-1], prediction))


# =============================================================================
# [3] END-TO-END ML PIPELINE (sklearn)
# =============================================================================
# Intuition:
#   A "pipeline" is the whole path from raw data → predictions:
#     load → split → scale → train (with hyperparam search) → evaluate.
#   Two golden rules:
#     1. NEVER touch test data during training. Fit the scaler on train only.
#     2. Use cross-validation for hyperparameter search, not the test set.
#
# FIXES applied vs. your original:
#   - `scaler.fit_transform(X_test)` → `scaler.transform(X_test)` (data leak).
#   - `"n_depth"` → `"n_estimators"`  (number of trees).
#   - `"min_sample_split"` → `"min_samples_split"`.
# -----------------------------------------------------------------------------
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# iris = load_iris()
# X = pd.DataFrame(iris.data, columns=iris.feature_names)
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y  # keep class balance
# )
#
# scaler = StandardScaler()          # (x - mean) / std, so features are ~N(0,1)
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)   # FIX: transform, not fit_transform
#
# model = RandomForestClassifier(random_state=42)
#
# param_grid = {
#     "n_estimators":      [50, 100, 200],   # more trees → lower variance
#     "max_depth":         [3, 5, 7],        # deeper → more capacity → overfit risk
#     "min_samples_split": [2, 5, 10],       # bigger → simpler trees
# }
#
# grid_search = GridSearchCV(
#     estimator=model, param_grid=param_grid,
#     cv=3, scoring="accuracy", n_jobs=-1,
# )
# grid_search.fit(X_train_scaled, y_train)
#
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# print("Best Parameters:", grid_search.best_params_)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))


# =============================================================================
# [4] CONV2D FROM SCRATCH
# =============================================================================
# Intuition:
#   A convolution slides a small window (the kernel) across the image and, at
#   each position, computes the element-wise product then sums it up.
#   This detects local patterns (edges, textures) regardless of position.
#
#   Output size formula:
#       out = (in - kernel + 2*padding) / stride + 1
#   Here we use no padding, so out = (in - kernel) / stride + 1.
# -----------------------------------------------------------------------------

# def conv2d(image, kernel, stride=1):
#     image_height, image_width  = image.shape
#     kernel_height, kernel_width = kernel.shape
#
#     output_height = (image_height - kernel_height) // stride + 1
#     output_width  = (image_width  - kernel_width)  // stride + 1
#
#     output = np.zeros((output_height, output_width))
#     for i in range(output_height):
#         for j in range(output_width):
#             # Pull out the patch under the kernel.
#             region = image[i*stride : i*stride + kernel_height,
#                            j*stride : j*stride + kernel_width]
#             # Element-wise multiply, then sum. This IS the dot product
#             # of flattened patch and flattened kernel — a similarity score.
#             output[i, j] = np.sum(region * kernel)
#     return output


# =============================================================================
# [5] SELF-ATTENTION FROM SCRATCH
# =============================================================================
# Intuition:
#   Every token asks: "which OTHER tokens should I pay attention to?"
#   We answer using three learned views of the same input:
#     Q (query):  what am I looking for?
#     K (key):    what do I contain?  (used to match against queries)
#     V (value):  what info do I actually pass along?
#
#   Score(i, j) = how well token i's query matches token j's key = Q_i · K_j
#   We divide by sqrt(d_k) to keep the numbers from blowing up when d_k is
#   large (dot products of many random values have variance ~ d_k, so
#   dividing by sqrt(d_k) keeps variance ~ 1, which keeps softmax stable).
#   Then softmax → attention weights (sum to 1 across the row).
#   Finally we mix the values: output_i = sum_j (weight_ij * V_j).
# -----------------------------------------------------------------------------
def softmax(x):
    # Subtract max for numerical stability: exp of huge numbers overflows.
    # Softmax is invariant to a constant shift, so this is safe.
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

    scores  = Q @ K.T / np.sqrt(d_k)   # (T, T) — how each token relates to every other
    weights = softmax(scores)          # rows sum to 1
    output  = weights @ V                # weighted average of values

    return output, weights             # FIX: original returned only output


# =============================================================================
# [6] BATCH NORMALIZATION
# =============================================================================
# Intuition:
#   Deep nets suffer from "internal covariate shift" — as we update earlier
#   layers, later layers keep seeing inputs with a different distribution.
#   BatchNorm forces each feature (across the batch) to have mean 0, variance 1,
#   then lets the network re-scale/shift via LEARNABLE gamma and beta.
#
#   Training:  use THIS batch's stats  (mean, var of current mini-batch)
#   Inference: use RUNNING stats accumulated during training
#              (because at test time you might have batch_size=1)
#
# Formulas:
#   x_hat = (x - mu_batch) / sqrt(var_batch + eps)     # normalize
#   y     = gamma * x_hat + beta                       # scale & shift
#   running_mean = momentum * running_mean + (1 - momentum) * mu_batch
#   running_var  = momentum * running_var  + (1 - momentum) * var_batch
#
# FIXES vs your original:
#   - epsilon default was 0.5 (huge). Should be ~1e-5.
#   - running_var update used batch_mean instead of batch_var.
# -----------------------------------------------------------------------------
# class BatchNorm:
#     def __init__(self, epsilon=1e-5, momentum=0.9):   # FIX epsilon
#         self.epsilon = epsilon
#         self.momentum = momentum
#         self.running_mean = 0
#         self.running_var  = 1
#         self.gamma = 1          # learnable scale
#         self.beta  = 0          # learnable shift
#
#     def forward(self, X, training=True):
#         if training:
#             batch_mean = np.mean(X, axis=0)
#             batch_var  = np.var(X,  axis=0)
#
#             # Normalize this batch to zero mean, unit variance.
#             X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
#             out    = self.gamma * X_norm + self.beta
#
#             # Exponential moving average — used at inference time.
#             self.running_mean = (self.momentum * self.running_mean +
#                                  (1 - self.momentum) * batch_mean)
#             self.running_var  = (self.momentum * self.running_var  +
#                                  (1 - self.momentum) * batch_var)   # FIX
#         else:
#             X_norm = ((X - self.running_mean) /
#                       np.sqrt(self.running_var + self.epsilon))
#             out = self.gamma * X_norm + self.beta
#         return out


# =============================================================================
# [7] ENCODER BLOCK (Transformer, PyTorch)
# =============================================================================
# Intuition:
#   One transformer encoder block = "look around, then think":
#     1. Self-attention lets each token look at every other token.
#     2. Feed-forward net processes each token independently ("thinking").
#   Each is wrapped in RESIDUAL + LAYER-NORM:
#       x = LayerNorm(x + Sublayer(x))
#   The residual lets gradients flow to early layers (no vanishing).
#   LayerNorm keeps activations from drifting.
#
# FIX vs original:
#   `nn.MultiheadAttention` returns a tuple (output, attention_weights).
#   You need to unpack it or index [0].
# -----------------------------------------------------------------------------
# import torch
# import torch.nn as nn
#
# class EncoderBlock(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             d_model, num_heads, dropout=dropout, batch_first=True
#         )
#         # FFN: expand → non-linearity → project back. Standard ratio is 4x.
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, d_model),
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x, attn_mask=None):
#         # Self-attention returns (out, weights) — take out.  FIX: added [0]
#         attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
#         x = self.norm1(x + self.dropout1(attn_out))    # residual + norm
#
#         ff_out = self.ffn(x)
#         x = self.norm2(x + self.dropout2(ff_out))      # residual + norm
#         return x


# =============================================================================
# [8] LINEAR REGRESSION FROM SCRATCH
# =============================================================================

class LinearRegression:
    """
    Linear regression trained via gradient descent.

    Model:
        y_pred = X @ w + b

    Loss (Mean Squared Error):
        L = (1/n) * sum( (y_pred - y)^2 )

    Gradients (why they look like this):
        dL/dw = (2/n) * X.T @ (y_pred - y)      # chain rule on the square
        dL/db = (2/n) * sum(y_pred - y)
        (The 2 is often absorbed into the learning rate — we drop it below.)

    Intuition:
        We fit the best straight line through the points. Gradient descent
        nudges w and b in the direction that reduces the squared error most.
        Squared error punishes big misses way more than small ones.
    """

    def __init__(self, learning_rate, num_iteration):
        self.lr = learning_rate
        self.iteration = num_iteration
        self.weights = None
        self.bias = None

    def fit(self, X, y):   # FIX: renamed from forward → fit (sklearn convention)
        """Train weights and bias with batch gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iteration):
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradient of MSE w.r.t. weights and bias.
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Step DOWNHILL on the loss surface.
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# =============================================================================
# [9] LOGISTIC REGRESSION FROM SCRATCH
# =============================================================================

class LogisticRegression:
    """
    Binary classification via logistic (sigmoid) function + gradient descent.

    Model:
        z      = X @ w + b
        y_pred = sigmoid(z) = 1 / (1 + exp(-z))     # squashes to (0, 1) — a probability

    Loss (Binary Cross-Entropy — NOT MSE):
        L = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )
        (BCE punishes confident-and-wrong predictions much harder than MSE.)

    Gradient magic:
        Despite the ugly loss, dL/dw simplifies BEAUTIFULLY to:
            dL/dw = (1/n) * X.T @ (y_pred - y)
            dL/db = (1/n) * sum(y_pred - y)
        (Same shape as linear regression! That's because sigmoid + BCE are
         "paired" to cancel out the messy chain-rule terms. Common interview
         question: derive this.)
    """

    def __init__(self, learning_rate, num_iterations):
        self.lr = learning_rate
        self.iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # FIX: your original was `1 / (1 + np.exp(z))` — sign was wrong.
        # Correct sigmoid is 1 / (1 + exp(-z)):
        #   z → +∞  gives  1  (very confident class 1)
        #   z →  0  gives  0.5 (uncertain)
        #   z → -∞  gives  0  (very confident class 0)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Same gradient shape as linear regression thanks to sigmoid+BCE pairing.
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        # Threshold at 0.5 — the decision boundary where p(class=1) = p(class=0).
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)


# =============================================================================
# [10] CNN TRAINING LOOP (PyTorch)
# =============================================================================
# Intuition:
#   Standard image-classification loop:
#     transforms → DataLoader → model → loss → optimizer.step() → repeat.
#
#   Key ideas:
#     - Convolutions extract local features (edges → textures → parts → objects
#       as we go deeper).
#     - MaxPool halves spatial size and adds translation invariance.
#     - CrossEntropyLoss expects raw logits (it does log-softmax internally).
#     - Adam adapts the learning rate per parameter automatically.
#
# FIX in your original: `def forward(self, X)` but body used lowercase `x`.
# -----------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
#
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),                          # HWC uint8 → CHW float [0, 1]
#     transforms.Normalize((0.5,), (0.5,)),           # then rescale to [-1, 1]
# ])
#
# train_ds = datasets.ImageFolder(root='path/to/train', transform=transform)
# test_ds  = datasets.ImageFolder(root='path/to/test',  transform=transform)
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)   # no shuffle at eval
#
# class CNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         # padding=1 with kernel=3 keeps spatial dims (SAME padding).
#         self.conv1 = nn.Conv2d(3,  16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H, W each time
#         # After 3 pools on 128x128 image: 128 → 64 → 32 → 16. So flatten = 64*16*16.
#         self.fc1 = nn.Linear(64 * 16 * 16, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)              # regularization on fc layer
#
#     def forward(self, x):                           # FIX: was `X` in signature
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)                   # flatten: (B, C, H, W) → (B, C*H*W)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x                                    # raw logits — CE loss adds softmax
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CNN(num_classes=len(train_ds.classes)).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# for epoch in range(10):
#     model.train()                                   # enables dropout, batchnorm updates
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()                       # gradients accumulate by default — clear them
#         output = model(images)
#         loss = criterion(output, labels)
#         loss.backward()                             # compute gradients via autograd
#         optimizer.step()                            # apply the update
#         running_loss += loss.item()
#     print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")
#
# model.eval()                                        # disables dropout, freezes batchnorm
# correct = total = 0
# with torch.no_grad():                               # skip building the autograd graph → faster
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         output = model(images)
#         _, predicted = torch.max(output, 1)         # argmax over classes
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy: {100 * correct / total:.2f}%')
#
# torch.save(model.state_dict(), 'cnn_model.pth')


# =============================================================================
# [11] CONV3D FROM SCRATCH (2D spatial conv over multi-channel input)
# =============================================================================
# NOTE: This is technically a 2D convolution over a multi-channel (RGB-like)
#       image, not a true 3D convolution over volumes. The kernel slides over
#       (H, W); the third dim is channels which we sum over.
#
# Padding:
#   pad = (kernel - 1) / 2  gives "SAME" padding → output same H, W as input.

def conv3d(image, kernel):
    """
    Multi-channel 2D convolution with SAME padding.

    Formula for each output pixel (i, j, k):
        out[i, j, k] = sum over (h, w, c) of  kernel[h, w, k] * padded[i+h, j+w, c]

    Intuition:
        Same as 2D conv, but each output channel k has its own filter that
        looks at ALL input channels and combines them. That's how a CNN mixes
        RGB or feature maps.
    """
    image_height, image_width, image_channels = image.shape
    kernel_height, kernel_width, kernel_channels = kernel.shape

    # SAME padding: keep output size == input size.
    pad_h = (kernel_height - 1) // 2
    pad_w = (kernel_width  - 1) // 2

    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )

    output_height   = image_height
    output_width    = image_width
    output_channels = kernel_channels

    output = np.zeros((output_height, output_width, output_channels))
    for i in range(output_height):
        for j in range(output_width):
            for k in range(output_channels):
                # For output channel k: element-wise multiply the k-th kernel
                # slice with the padded image patch (summing across ALL input channels).
                output[i, j, k] = np.sum(
                    kernel[:, :, k] * padded_image[i:i+kernel_height,
                                                   j:j+kernel_width, :]
                )
    return output


# =============================================================================
# [12] DECISION TREE CLASSIFIER FROM SCRATCH
# =============================================================================

class Node:
    """A single node in the tree. Leaves store `value` (predicted class)."""
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature      # which column to split on
        self.threshold = threshold    # split point
        self.left      = left         # subtree for values < threshold
        self.right     = right        # subtree for values >= threshold
        self.value     = value        # None for internal, class label for leaves


def gini(y):
    """
    Gini impurity: probability of misclassifying a random sample if we
    labelled it by picking a class proportional to the class frequencies.

    Formula:  gini = 1 - sum_c p_c^2

    Intuition:
        - All samples same class → p = 1 for that class → gini = 0 (pure).
        - Two classes 50/50    → gini = 1 - 0.5 = 0.5 (max messy for binary).
        Lower = purer = better.

    Alternative: entropy = -sum p_c * log(p_c). Gini and entropy behave very
    similarly; gini is a bit faster (no log).
    """
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity


def split(X, y, feature, threshold):
    """Split rows by whether X[:, feature] < threshold."""
    left_mask  = X[:, feature] < threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


def best_split(X, y):
    """
    Try every (feature, threshold) combo; keep the one that gives the
    LOWEST weighted-average gini after the split.

    Weighted formula:
        g = (|L|*gini(L) + |R|*gini(R)) / |parent|
    (Bigger children contribute more — a tiny pure leaf shouldn't dominate.)
    """
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_X, right_X, left_y, right_y = split(X, y, feature, threshold)
            if len(left_X) == 0 or len(right_X) == 0:
                continue          # useless split, skip

            g = (len(left_y) * gini(left_y) +
                 len(right_y) * gini(right_y)) / len(y)

            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


def most_common(y):
    """Majority-class label — used for leaf predictions."""
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def build_tree(X, y, depth=0, max_depth=None):
    """
    Recursively grow the tree. Stopping conditions:
        1. All labels are the same (pure node → make a leaf).
        2. Max depth reached (prevent overfitting).
        3. No useful split found (all features constant).
    """
    if len(np.unique(y)) == 1:
        return Node(value=y[0])                        # pure

    if max_depth is not None and depth >= max_depth:
        return Node(value=most_common(y))              # depth limit

    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=most_common(y))              # no good split

    left_X, right_X, left_y, right_y = split(X, y, feature, threshold)
    left_tree  = build_tree(left_X,  left_y,  depth + 1, max_depth)
    right_tree = build_tree(right_X, right_y, depth + 1, max_depth)
    return Node(feature, threshold, left_tree, right_tree)


def predict_one(node, x):
    """Walk the tree from root to leaf following the splits."""
    if node.value is not None:
        return node.value
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    return predict_one(node.right, x)


class DecisionTreeClassifier:
    """
    Full API. Fit builds the tree; predict walks each row through it.

    Big picture:
        Trees keep splitting the data into purer and purer subsets. Each
        split is a "yes/no question" about one feature. Leaves are the
        answers (a class label).

    Pros: interpretable, no scaling needed, handles mixed feature types.
    Cons: high variance (small data change → very different tree) →
          that's why we combine many trees into Random Forests / GBDTs.
    """
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = build_tree(X, y, max_depth=self.max_depth)

    def predict(self, X):
        return np.array([predict_one(self.root, row) for row in X])


# =============================================================================
# [13] FEEDFORWARD NN WITH BACKPROP FROM SCRATCH (Linear + GeLU + FFN)
# =============================================================================

class Linear:
    """
    A fully-connected (dense) layer with SGD update built in.

    Forward:
        y = X @ W + B

    Backward (chain rule):
        Given dL/dy (grad_output from upstream):
            dL/dW = X.T @ dL/dy         # shapes: (in, N) @ (N, out) = (in, out)
            dL/dB = sum over batch of dL/dy
            dL/dX = dL/dy @ W.T         # what gets passed back to earlier layers

    Intuition:
        Every weight controls how much one input feature contributes to one
        output. dL/dW says "if I nudge this weight up, does loss go up or
        down, and by how much?" Then we step against that direction.

    Init note:
        `W = randn * 0.01` is a small-init trick to keep early activations
        from exploding. For deeper nets, prefer Xavier/Kaiming (below).
    """
    def __init__(self, in_features, out_features, learning_rate: float = 0.0001):
        self.learning_rate = learning_rate
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.B = np.zeros(out_features)
        # Cached during forward, used during backward:
        self.x = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        self.x = X                                    # cache for backward
        return X @ self.W + self.B

    def backward(self, grad_output):
        self.dW = self.x.T @ grad_output              # (in, N) @ (N, out) = (in, out)
        self.dB = np.sum(grad_output, axis=0)         # sum grads over the batch
        grad_input = grad_output @ self.W.T           # pass grad to previous layer
        return grad_input

    def step(self):
        """One vanilla SGD update: W ← W - lr * dW."""
        self.W -= self.learning_rate * self.dW
        self.B -= self.learning_rate * self.dB

def sigmoid(x):
    """
    Sigmoid activation function.

    Formula:
        sigmoid(x) = 1 / (1 + exp(-x))

    Intuition:
        - Converts any real number into the range (0, 1).
        - Large positive values become close to 1.
        - Large negative values become close to 0.
        - Commonly used for binary classification because its output
          can be interpreted as a probability.
    """
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    """
    Sigmoid activation layer.

    Forward:
        y = 1 / (1 + exp(-x))

    Derivative:
        d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))

    Intuition:
        - The gradient is largest around x = 0.
        - As x becomes very positive or very negative,
          the gradient approaches zero.
        - During backpropagation, we multiply the upstream
          gradient by this derivative (chain rule).
    """

    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        dsigmoid = self.output * (1 - self.output)
        return grad_output * dsigmoid


class FeedForwardNetwork:
    """
    The transformer FFN sub-block:
        Linear (d_model → hidden)  →  GeLU  →  Linear (hidden → d_model)

    Intuition:
        Attention mixes information ACROSS tokens.
        FFN processes each token INDIVIDUALLY, expanding to a bigger space
        (usually 4x d_model) so more complex features can be learned before
        projecting back.
    """
    def __init__(self, d_model, hidden_dim):
        self.linear1 = Linear(d_model, hidden_dim)
        self.sigmoid    = Sigmoid()
        self.linear2 = Linear(hidden_dim, d_model)

    def forward(self, X):
        x = self.linear1.forward(X)
        x = self.sigmoid.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, grad):
        # Backprop = walk forward pass IN REVERSE, passing gradients back.
        grad = self.linear2.backward(grad)
        grad = self.sigmoid.backward(grad)
        grad = self.linear1.backward(grad)

    def step(self):
        self.linear1.step()
        self.linear2.step()


def mse_loss(pred, target):
    """L = mean( (pred - target)^2 )"""
    return np.mean((pred - target) ** 2)


def mse_backward(pred, target):
    """
    dL/dpred = 2 * (pred - target) / N
    where N = total number of elements (that's what pred.size gives us).
    """
    return (2 * (pred - target)) / pred.size


# =============================================================================
# [14] MULTI-LAYER FEEDFORWARD NN WITH TRAINING (from scratch)
# =============================================================================

# --- Activations ---------------------------------------------------------------
def relu(x):
    """ReLU: max(0, x). Kills negatives → sparse, cheap, works great in practice."""
    return np.maximum(0, x)


def relu_derivative(x):
    """dReLU/dx = 1 if x > 0 else 0. (Zero at x=0 is a chosen convention.)"""
    return (x > 0).astype(float)


def sigmoid(x):
    """1 / (1 + e^{-x}). Squashes to (0, 1) → useful as a probability."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    dSigmoid/dx = sigmoid(x) * (1 - sigmoid(x))
    Peaks at 0.25 when x=0, drops toward 0 for large |x| → causes
    vanishing gradients in deep nets (that's why ReLU took over).
    """
    s = sigmoid(x)
    return s * (1 - s)


# --- Losses --------------------------------------------------------------------
# Note: `mse_loss` is redefined here to keep both blocks self-contained.
def mse_loss(predicted, actual):  # noqa: F811
    """MSE: average of squared errors. Good for regression."""
    errors = predicted - actual
    return np.mean(errors ** 2)

def binary_cross_entropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    # y_true: true labels (0 or 1)
    # y_pred: predicted probabilities
    # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
    # return round(your_answer, 4)
    epsilon = 1e-4
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return round(loss, 4)

def categorical_cross_entropy( y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
    # y_pred: predicted probabilities (shape: n_samples x n_classes)
    # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
    # return round(your_answer, 4)
    epsilon = 1e-4

    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.mean(np.sum(y_true * (1 - y_pred), axis=1))

    return round(loss, 4)


def binary_cross_entropy_loss(predicted, actual):
    """
    BCE = -mean( y*log(p) + (1-y)*log(1-p) )

    Intuition:
        If y=1 and p=0.99 → loss ~ 0 (great).
        If y=1 and p=0.01 → loss ~ 4.6 (huge). Punishes confident wrong preds.

    We clip probabilities away from 0 and 1 to avoid log(0) = -inf.
    """
    epsilon = 1e-8
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    return -np.mean(actual * np.log(predicted) +
                    (1 - actual) * np.log(1 - predicted))


class SimpleNeuralNetwork:
    """
    Fully-connected multi-layer network with backprop.

    Architecture:
        input → [hidden1 → hidden2 → ...] → output
        Each connection is a matrix multiply followed by an activation.

    Init: `W = randn / sqrt(fan_in)` — Xavier-like init that keeps the
          variance of activations roughly constant across layers.

    Regularization (L2 / weight decay):
        loss_total = loss_data + 0.5 * lambda * sum(W^2)
        d(reg)/dW  = lambda * W
        Adds a penalty for large weights → forces the model to prefer
        simpler (smaller-weight) solutions → less overfitting.

    Training loop:
        for each epoch:
            pred = forward(X)
            grads = backward(X, y, pred)
            update W, B using SGD
    """

    def __init__(self, input_size, hidden_size, output_size,
                 activation="sigmoid", regularization_strength: float = 0.0,
                 loss_function=None):
        # Full list of layer widths: [in, h1, h2, ..., out]
        self.layer_sizes = [input_size] + hidden_size + [output_size]
        self.num_connections = len(self.layer_sizes) - 1

        # Pick activation and its derivative.
        if activation == 'sigmoid':
            self.activate = sigmoid
            self.activate_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activate = relu
            self.activate_derivative = relu_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.loss_function = loss_function if loss_function else mse_loss

        # Xavier-style initialization: divide by sqrt(fan_in).
        self.weights = []
        self.biases  = []
        for i in range(self.num_connections):
            neurons_in  = self.layer_sizes[i]
            neurons_out = self.layer_sizes[i + 1]
            W = np.random.randn(neurons_in, neurons_out) / np.sqrt(neurons_in)
            b = np.zeros(neurons_out)
            self.weights.append(W)
            self.biases.append(b)

        self.reg_strength = regularization_strength

    def forward(self, X):
        """
        Push X through every layer, caching the raw scores (pre-activation)
        and outputs (post-activation) so we can use them in backward().
        """
        self.layer_inputs  = []      # pre-activation values (z)
        self.layer_outputs = [X]     # post-activation values (a); index 0 = input

        current_data = X
        for i in range(self.num_connections):
            raw_scores = current_data @ self.weights[i] + self.biases[i]
            self.layer_inputs.append(raw_scores)
            activated  = self.activate(raw_scores)
            self.layer_outputs.append(activated)
            current_data = activated
        return current_data

    def backward(self, X, y, prediction):
        """
        Backprop, layer by layer, from output to input.

        Output-layer error signal (delta):
            delta_L = (pred - y) / N * activation'(z_L)
            (This assumes MSE loss + activation at output. For BCE + sigmoid
             at the output, the two derivatives cancel and delta_L = (pred - y)/N.
             We use the general form to stay flexible.)

        For each earlier layer:
            dL/dW_i = a_{i-1}.T @ delta_i        # matches shape (in, out)
            dL/db_i = sum over batch of delta_i
            delta_{i-1} = (delta_i @ W_i.T) * activation'(z_{i-1})   # push error back

        L2 reg adds `reg_strength * W_i` to dL/dW_i (derivative of 0.5*λ*W^2).
        """
        num_samples = X.shape[0]

        # Start error signal at the OUTPUT.
        output_raw_scores = self.layer_inputs[-1]
        error_signal = ((prediction - y) / num_samples *
                        self.activate_derivative(output_raw_scores))

        weight_gradients = [None] * self.num_connections
        bias_gradients   = [None] * self.num_connections

        for i in reversed(range(self.num_connections)):
            prev_output = self.layer_outputs[i]                  # activation from layer i-1
            dW = prev_output.T @ error_signal + self.reg_strength * self.weights[i]
            db = np.sum(error_signal, axis=0)
            weight_gradients[i] = dW
            bias_gradients[i]   = db

            # Push the error signal one layer earlier (chain rule).
            if i > 0:
                prev_raw_scores  = self.layer_inputs[i - 1]
                prev_sensitivity = self.activate_derivative(prev_raw_scores)
                error_signal = (error_signal @ self.weights[i].T) * prev_sensitivity

        return weight_gradients, bias_gradients

    def compute_loss(self, prediction, actual):
        """Data loss + weight penalty (L2)."""
        prediction_error = self.loss_function(prediction, actual)
        weight_penalty = 0.0
        for W in self.weights:
            weight_penalty += 0.5 * self.reg_strength * np.sum(W ** 2)
        return prediction_error + weight_penalty

    def train(self, X, y, epochs=1000, learning_rate=0.01, print_every=100):
        for epoch in range(epochs):
            prediction = self.forward(X)
            weight_grads, bias_grads = self.backward(X, y, prediction)

            # Vanilla SGD update on all layers.
            for i in range(self.num_connections):
                self.weights[i] -= learning_rate * weight_grads[i]
                self.biases[i]  -= learning_rate * bias_grads[i]

            if epoch % print_every == 0:
                loss = self.compute_loss(prediction, y)
                print(f"  Epoch {epoch:>4d}  |  Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def predict_classes(self, X, threshold=0.5):
        """For binary: probability ≥ 0.5 → class 1, else class 0."""
        return (self.predict(X) >= threshold).astype(int)


# =============================================================================
# [15] K-MEANS CLUSTERING
# =============================================================================

def assign_clusters(X, centroids):
    """
    For each point, find the nearest centroid.

    Uses broadcasting:
        X[:, None, :]         → shape (N, 1, D)
        centroids[None, :, :] → shape (1, K, D)
        diffs                 → shape (N, K, D)   (N points × K centroids)
        dists[n, k]           = distance from point n to centroid k
    argmin over centroids gives each point's cluster label.
    """
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    return np.argmin(dists, axis=1)


def update_centroids(X, label, k):
    """
    New centroid = mean of all points assigned to that cluster.

    Empty-cluster case: if no point was assigned to cluster i, re-seed it
    with a random point (otherwise the centroid gets stuck at the origin).

    FIX vs your original:
        `X[np.random.randn(len(X))]` was buggy — randn returns floats.
        Should be `X[np.random.randint(len(X))]`.
    """
    new = np.zeros((k, X.shape[1]))
    for i in range(k):
        pts = X[label == i]
        if len(pts) == 0:
            new[i] = X[np.random.randint(len(X))]   # FIX
        else:
            new[i] = pts.mean(axis=0)
    return new


def inertia(X, labels, centroids):
    """
    Inertia (a.k.a. WCSS — Within-Cluster Sum of Squares):
        sum over clusters of  sum over points-in-cluster of ||x - centroid||^2

    Intuition: how tightly packed are points around their centroid?
    Lower = tighter clusters. K-means minimizes exactly this.
    Used to pick k via the ELBOW method: plot inertia vs k, look for the
    "elbow" where the drop flattens out.
    """
    return float(
        sum( # overall sum of all centroids
            (
                    (X[labels == i] - centroids[i]) ** 2
            ).sum() # each cluster distance from centroid and then summed up for each cluster
                     for i in range(len(centroids))
        )
    )


def kmeanspp_init(X, k, rng):
    """
    K-means++ initialization: spread starting centroids apart to avoid the
    "all centroids land in one blob" failure of naive random init.

    Algorithm:
        1. Pick first centroid uniformly at random.
        2. For each subsequent centroid, sample a point with probability
           proportional to its SQUARED distance from the nearest existing
           centroid. Faraway points are more likely to be picked.
    """
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        # For each point, squared distance to nearest current centroid.
        d2 = np.min(
            [
                ((X - c) ** 2).sum(axis=1) for c in centroids
            ],
            axis=0)
        probs = d2 / d2.sum()
        centroids.append(X[rng.choice(len(X), p=probs)])
    return np.array(centroids, dtype=float)


def random_init(X, k, rng):
    """Naive random init: pick k random points as centroids. FIX: replce → replace."""
    return X[rng.choice(len(X), k, replace=False)].astype(float)   # FIX typo


def kmeans(X, k, max_iters=100, tol: float = 1e-4, seed=0,
           init="kmeans++", n_init=10):
    """
    Full K-means with multiple restarts.

    Algorithm (per restart):
        repeat:
            labels = assign each point to nearest centroid
            new_centroids = mean of points per cluster
            if centroids barely moved → stop
        record final inertia
    Keep the run with the LOWEST inertia (K-means is non-convex; different
    inits give different results).

    Complexity: O(n_init * max_iters * N * K * D).
    """
    best = None
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centroids = kmeanspp_init(X, k, rng)  # (use random_init here to compare)

        for iter in range(max_iters):
            labels = assign_clusters(X, centroids)
            new_centroids = update_centroids(X, labels, k)
            # Total centroid movement — if tiny, we've converged.
            shift = np.sqrt(((new_centroids - centroids) ** 2).sum())
            centroids = new_centroids
            if shift < tol:
                break

        score = inertia(X, labels, centroids)
        if best is None or score < best[0]:
            best = (score, labels, centroids, iter + 1)

    score, labels, centroids, iters = best
    return labels, centroids, score, iters


def elbow_scan(X, k_range, **kw):
    """Try many k values; return {k: inertia} for the elbow plot."""
    return {k: kmeans(X, k, **kw)[2] for k in k_range}


# =============================================================================
# [16] PERCEPTRON
# =============================================================================

class Perceptron:
    """
    The original neural unit (Rosenblatt, 1958). Binary classifier.

    Model:
        y_hat = 1 if (X @ w + b) >= 0 else 0

    Update rule (only when a mistake is made):
        w ← w + lr * (y - y_hat) * x
        b ← b + lr * (y - y_hat)

    Intuition:
        - If we predicted 0 but truth is 1: nudge the decision boundary
          TOWARD this point (add lr*x to w).
        - If we predicted 1 but truth is 0: nudge AWAY (subtract).
        - If correct: don't touch anything. This is the "mistake-driven"
          learning that makes it so simple.

    Big caveat:
        The perceptron only converges if the data is LINEARLY SEPARABLE.
        On XOR (classic counter-example), it loops forever with no line
        that can separate the classes. This "no non-linear boundary" limit
        is what motivated multi-layer nets.
    """
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def net_input(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.net_input(X) >= 0).astype(int)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.mistakes_per_epoch = []

        for epoch in range(self.epochs):
            mistakes = 0
            # One sample at a time (online / stochastic update).
            for xi, yi in zip(X, y):
                y_hat = 1 if (xi @ self.w + self.b) >= 0 else 0
                error = yi - y_hat
                if error != 0:                       # only update on mistakes
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    mistakes += 1
            self.mistakes_per_epoch.append(mistakes)
            if mistakes == 0:                        # perfect epoch → converged
                break

        self.epochs_used = epoch + 1
        self.converged = (mistakes == 0)
        return self



class LinearSVM:
    """
    Soft-margin linear SVM trained via subgradient descent on hinge loss.

    Loss:
        L(w) = 0.5 * ||w||^2  +  C * sum_i max(0, 1 - y_i * (w·x_i + b))
                ^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                margin term       hinge loss (only misclassified/margin-violating points)

    Intuition:
        Find the widest possible "street" (margin) between the two classes.
        C controls the trade-off:
            - Big C → punish misclassifications heavily → narrower margin, low bias.
            - Small C → allow more mistakes → wider margin, more regularization.

    Update rule (per point):
        margin_i = y_i * (w·x_i + b)
        if margin_i >= 1:    # point is safely OUTSIDE the margin
            w ← w - lr * w                            # just L2 shrinkage
        else:                                          # margin violated
            w ← w - lr * (w - C * y_i * x_i)          # shrinkage + push
            b ← b + lr * C * y_i

    FIX vs your original:
        `y = np.where(y >= 0, 1, -1)` mapped 0 → 1, which is wrong if labels
        arrive as 0/1. We now handle both {0, 1} and {-1, 1} explicitly.
    """
    def __init__(self, lr: float = 0.01, epochs: int = 100, C: float = 1.0):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize labels to {-1, +1} regardless of input encoding.
        # (Both {0, 1} and {-1, 1} → {-1, +1}.)
        y = np.where(y <= 0, -1, 1)   # FIX

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                x_i = X[i]
                # Signed distance × label. > 1 means "correctly classified
                # AND outside the margin". <= 1 means margin violation.
                margin = y[i] * (np.dot(x_i, self.weights) + self.bias)

                if margin >= 1:
                    # Only the regularization gradient acts on w.
                    self.weights -= self.lr * self.weights
                else:
                    # Regularization + push toward correct side.
                    self.weights -= self.lr * (
                        self.weights - self.C * y[i] * x_i
                    )
                    self.bias += self.lr * self.C * y[i]

    def decision_function(self, X):
        """Raw score (signed distance to the hyperplane, up to ||w||)."""
        return np.dot(X, self.weights) + self.bias

    def prediction(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)



def get_minimizer(iterations: int, learning_rate: float, init: int) -> float:
    # Objective function: f(x) = x^2
    # Derivative:         f'(x) = 2x
    # Update rule:        x = x - learning_rate * f'(x)
    # Round final answer to 5 decimal places
    x = init

    for _ in range(iterations):
        gradient = 2 * x
        x = x - learning_rate * gradient

    return round(x, 5)

def sigmoid_neetcode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
    # z is a 1D NumPy array
    # Formula: 1 / (1 + e^(-z))
    # return np.round(your_answer, 5)
    return np.round(1 / (1 + np.exp(-z)), 5)

def relu_neetcode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
    # z is a 1D NumPy array
    # Formula: max(0, z) element-wise
    return np.round(np.maximum(0, z), 5)

# if __name__ == "__main__":
#
#     # -------- Linear SVM demo --------
#     X = np.array([[2, 3], [3, 3], [2, 1],
#                   [7, 8], [8, 8], [9, 10]])
#     y = np.array([-1, -1, -1, 1, 1, 1])
#
#     svm = LinearSVM(lr=0.001, C=1.0, epochs=5000)
#     svm.fit(X, y)
#
#     print("Weights :", svm.weights)
#     print("Bias    :", svm.bias)
#     predictions = svm.prediction(X)
#     print("\nPredictions:")
#     print(predictions)
#     accuracy = np.mean(predictions == y)
#     print(f"\nAccuracy: {accuracy:.2f}")


# =============================================================================
# =============================================================================
# MISSING TOPICS — STUDY LIST (high-priority for MLE interviews)
# =============================================================================
# =============================================================================
#
# CORE (very frequently asked at FAANG / top companies):
# ------------------------------------------------------
#  1. Softmax + Cross-Entropy loss (with numerically-stable log-sum-exp)
#     - Derive that d(softmax + CE)/d(logits) = (p - y). Classic whiteboard Q.
#
#  2. Multi-Head Attention from scratch (extension of your single-head).
#     - Split d_model into H heads, run attention in parallel, concat, project.
#
#  3. Positional Encoding
#     - Sinusoidal (original Transformer)
#     - Learned positional embeddings
#     - RoPE (Rotary — used in LLaMA, GPT-NeoX, most modern LLMs)
#
#  4. Layer Normalization from scratch (you have BatchNorm — different axis).
#     - LN normalizes across FEATURES per sample; BN across BATCH per feature.
#
#  5. Dropout from scratch
#     - Zero out fraction p of activations at train time; scale by 1/(1-p).
#
#  6. Optimizers from scratch
#     - SGD → Momentum → Nesterov → RMSProp → Adam → AdamW.
#     - Know the Adam update rule cold: m_t, v_t, bias correction.
#
#  7. PCA from scratch
#     - Center → covariance → eigendecomposition (or SVD) → top-k components.
#
#  8. Naive Bayes classifier
#     - Bayes rule + conditional independence assumption. Gaussian / Multinomial.
#
#  9. Random Forest (bagging on top of your decision tree!)
#     - Bootstrap samples + random feature subset per split. Reduces variance.
#
# 10. Gradient Boosting (conceptual + XGBoost)
#     - Fit tree_n to the residuals of tree_{n-1}. Boosting = bias reduction.
#
# 11. K-Fold Cross-Validation from scratch.

#
# LLM / MODERN:
# -------------
# 21. LoRA (Low-Rank Adaptation) — parameter-efficient finetuning.
#     - W_new = W + A @ B where A, B are low-rank.
#
# 22. KV cache (why LLM inference is O(n) instead of O(n^2) per token).
#
# 23. Speculative decoding / draft models (fast LLM inference).
#
# 24. RLHF / DPO basics (conceptual — reward model + PPO or preference loss).
#
# 25. Retrieval-Augmented Generation (RAG) — embed, retrieve, augment prompt.
#
# 26. Quantization: INT8 / INT4 basics, GPTQ / AWQ conceptually.
#
# LOSSES / METRICS WORTH KNOWING:
# -------------------------------
# 27. Contrastive loss / Triplet loss / InfoNCE (SimCLR, CLIP-style).
# 28. Focal Loss (for class imbalance).
# 29. Huber / Smooth-L1 loss (robust regression).
# 30. Precision / Recall / F1 / ROC-AUC — implement from scratch.
#
# PRACTICAL SYSTEM DESIGN (common in system-design rounds):
# --------------------------------------------------------
# 31. Design a recommender system (candidate gen + ranking).
# 32. Design a feed ranking / ads CTR prediction system.
# 33. Design a nearest-neighbour search at scale (ANN: HNSW, IVF, PQ).
# 34. Feature store design + train/serve skew.
# 35. Online learning / drift detection.
#
#
# =============================================================================
# LIKELY FOLLOW-UP QUESTIONS FROM CODE YOU'VE ALREADY WRITTEN
# =============================================================================
#
# On KNN:
#   - What's the time complexity? How to speed it up? (KD-tree, Ball tree, LSH,
#     FAISS/HNSW for high-dim.)
#   - Why does KNN fail in high dimensions? (Curse of dimensionality — distances
#     concentrate; every point becomes roughly equidistant.)
#   - Weighted KNN: weight votes by 1/distance.
#   - How to choose k? Cross-validation. Small k → high variance (overfit),
#     large k → high bias (underfit).
#   - Why standardize features before KNN?
#
# On Linear / Logistic Regression:
#   - Derive the closed-form solution for linear regression: w = (X.T X)^-1 X.T y.
#   - When does it fail? (Singular X.T X — collinear features. Solutions: ridge,
#     pseudo-inverse.)
#   - Derive the logistic regression gradient (from BCE + sigmoid).
#   - Why does LR use BCE loss, not MSE? (MSE + sigmoid → non-convex; BCE → convex.)
#   - Multiclass extension → softmax regression.
#   - L1 vs L2 regularization: L1 → sparse weights; L2 → small weights.
#
# On Conv2D/3D:
#   - Compute output shape given input, kernel, stride, padding, dilation.
#   - Number of parameters in a conv layer: in_ch * out_ch * kH * kW + out_ch.
#   - Receptive field growth as you stack layers.
#   - Depthwise separable convolutions (MobileNet).
#   - Why pooling? Translation invariance + downsampling.
#
# On Self-Attention:
#   - Why divide by sqrt(d_k)?  (Keeps variance ~1 so softmax stays sharp/stable.)
#   - Complexity? O(T^2 * d) — quadratic in sequence length. Solutions:
#     linear attention, sparse attention (Longformer), FlashAttention (I/O optimal).
#   - Difference between self- and cross-attention (Q from one source, K,V from another).
#   - Causal masking (upper-triangular -inf) for autoregressive models.
#
# On BatchNorm:
#   - When does BN fail? Very small batch sizes → noisy stats. Use LayerNorm.
#   - BN placement: before or after activation? (Usually before — arguable.)
#   - BN during inference: use running stats, not batch stats.
#   - BN vs LN vs GroupNorm vs InstanceNorm — know the axes each normalizes over.
#
# On Encoder Block:
#   - Why LayerNorm not BatchNorm in transformers? (Variable seq lengths; small
#     effective batches per token.)
#   - Pre-norm vs post-norm — pre-norm trains more stably for deep transformers.
#   - Why is FFN hidden dim usually 4 * d_model?
#
# On CNN training:
#   - Difference between model.train() and model.eval().
#   - Why zero_grad() every step?
#   - Explain torch.no_grad() during evaluation.
#   - Learning rate schedules (step, cosine, warmup + decay).
#
# On Decision Tree:
#   - Gini vs entropy — practical difference? (Almost none; gini is faster.)
#   - How to handle continuous features? (Try midpoints of sorted unique values.)
#   - How to handle categorical features?
#   - How does a tree overfit, and how do we prevent it? (Depth limit, min samples,
#     pruning, ensembling.)
#   - Extend to Random Forest → Gradient Boosted Trees (XGBoost).
#
# On Feedforward / Multi-layer NN:
#   - Derive backprop for one layer on paper.
#   - Vanishing / exploding gradients — causes and fixes (ReLU, LayerNorm,
#     residuals, better init).
#   - Xavier vs Kaiming init — when to use which. (Xavier for tanh/sigmoid,
#     Kaiming for ReLU.)
#   - Why residual connections help. (Identity path lets gradients flow.)
#   - L1 vs L2 regularization behavior on weights.
#
# On K-Means:
#   - Complexity? O(N * K * D * iters).
#   - Why k-means++ over random init?
#   - How to pick k? (Elbow method, silhouette score, gap statistic.)
#   - Failure modes: non-spherical clusters, different cluster sizes, outliers.
#     → Try DBSCAN, Gaussian Mixture Models, spectral clustering.
#   - Is K-means guaranteed to converge? (Yes to a local min — that's why n_init.)
#
# On Perceptron:
#   - Why can't a single perceptron learn XOR?
#   - Perceptron convergence theorem: guaranteed convergence in finite steps
#     if data is linearly separable.
#   - Perceptron vs Logistic Regression vs SVM — three views of the same idea
#     with different loss functions.
#
# On Linear SVM:
#   - Derive the hinge loss and its subgradient.
#   - What are support vectors? Points on or inside the margin.
#   - Effect of C (soft-margin trade-off).
#   - Kernel trick — how does SVM handle non-linear boundaries without going
#     into feature space explicitly? (RBF kernel, polynomial kernel.)
#   - SVM vs Logistic Regression: hinge loss vs log loss; SVM cares only about
#     margin violators, LR uses every point.
#
# =============================================================================



# =============================================================================
# MULTI HEAD SELF ATTENTION
# =============================================================================

class MultiHeadSelfAttention:

    def __init__(self, d_model : int, num_heads: int):

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.scale = np.sqrt(d_k)

        self.Wq = np.random.randn(d_model, d_model) / self.scale
        self.Wk = np.random.randn(d_model, d_model) / self.scale
        self.Wv = np.random.randn(d_model, d_model) / self.scale
        self.Wo = np.random.randn(d_model, d_model) / self.scale


    def softmax(self, x):
        np_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return np_exp / np.sum(np_exp, axis=-1, keepdims=True)


    def split(self, X):

        B, L, _ = X.shape

        x = X.reshape(B, L, self.num_heads, self.d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        return x

    def combine(self, X):
        B, L, H, d_k = X.shape
        x = np.transpose(X, (0, 2, 1, 3))
        x = x.reshape(B, L, H * d_k)
        return x

    def scaled_self_attention(self, Q, K, V):
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        scores /= np.sqrt(self.d_k)

        weights = self.softmax(scores)

        output = np.matmul(weights, V)
        return output

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        Q = self.split(Q)
        K = self.split(K)
        V = self.split(V)

        attention = self.scaled_self_attention(Q, K, V)

        concat = self.combine(attention)

        output = np.matmul(concat, self.Wo)

        return output





# =============================================================================
# SINUSODIAL POSITIONAL ENCODING
# =============================================================================

import numpy as np


def positional_encoding(max_seq_len: int, d_model: int):
    pe = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            denominator = 10000 ** (i / d_model)
            pe[pos, i] = np.sin(pos / denominator)
            if i + 1 < max_seq_len:
                pe[pos, i + 1] = np.cos(pos / denominator)
    return pe

pe = positional_encoding(
    max_seq_len=5,
    d_model=8
)
print(np.round(pe, 4))

