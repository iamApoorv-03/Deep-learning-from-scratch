# Deep Learning Foundations: From Scratch to Class

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Status](https://img.shields.io/badge/Status-Active_Development-green?style=for-the-badge)

---

# Overview

This repository documents a structured journey of learning **Deep Learning from first principles using PyTorch**.

Instead of treating neural networks as black boxes, this project focuses on:

- Understanding **gradient-based optimization**
- Implementing models **step-by-step from scratch**
- Studying **training dynamics through experiments**
- Building **professional deep learning training pipelines**

The project evolves through progressively complex stages:

```
Linear Regression → Neural Networks → Training Pipelines → Optimization → Classification
```

Each stage introduces both **theoretical concepts** and **practical implementations**.

---

# Learning Philosophy

The main philosophy of this repository is:

```
Concept → Implementation → Experiment → Observation
```

Every concept is:

- implemented in code
- experimentally validated
- analyzed through observations

This approach helps develop **true intuition for deep learning systems**.

---

# Evolution Phases

---

# Phase 1 — Linear Regression From Scratch

### Concepts

- Gradient Descent
- Mean Squared Error (MSE)
- Learning Rate behavior

### Mechanics

- `requires_grad=True`
- `loss.backward()`
- manual weight updates
- gradient zeroing

### Key Insight

Optimization stability strongly depends on the **learning rate**.

Large learning rates cause **divergence**, while small learning rates slow convergence.

---

# Phase 2 — Noise Experiments & Bias–Variance Tradeoff

### Experiment

Synthetic dataset:

```
y = 3x² + 2x + 3 + noise
```

Gaussian noise was added to analyze:

- parameter stability
- irreducible error
- model variance

### Key Insight

Noise increases **irreducible error** and demonstrates the **bias–variance tradeoff**.

Models can begin **memorizing noise instead of learning the underlying function**.

---

# Phase 3 — Polynomial Regression

### Feature Engineering

Polynomial features were added:

$$
X_poly = [x , x²]
$$

### Feature Normalization

$$
X_norm = (X - mean) / std
$$

### Key Insight

Feature scaling is essential for stable optimization.

Without normalization, gradients can **explode due to skewed feature magnitudes**.

---

# Phase 4 — Neural Networks

First neural network architecture:

```python
nn.Sequential(
    nn.Linear(1,32),
    nn.ReLU(),
    nn.Linear(32,1)
)
```

### Concepts

- nonlinear activation functions
- universal function approximation
- piecewise linear modeling

### Key Insight

ReLU allows neural networks to approximate **nonlinear relationships** by stacking linear transformations.

---

# Phase 5 — Mini-Batch Training (DataLoader)

### Implementation

- `TensorDataset`
- `DataLoader`

Training now processes **mini-batches**:

```python
for batch_X, batch_Y in train_loader:
```

### Key Insight

Mini-batch training introduces **stochastic gradient noise**, which often helps optimization escape poor local minima.

Tracking **average epoch loss** produces more stable diagnostics.

---

# Phase 6 — Regularization (L2 & Dropout)

### Techniques Implemented

- Weight decay (L2 regularization)
- Dropout layers

### Key Insight

Regularization reduces **variance** but increases **bias**.

It becomes important when **model capacity exceeds dataset complexity**.

---

# Phase 7 — Overfitting Diagnosis

### Experiments

Overfitting scenarios were created by:

- increasing model capacity
- reducing dataset size
- increasing noise

### Observation

```
training loss ↓
evaluation loss ↑
```

This indicates **overfitting**, where the model memorizes noise.

### Insight

Model complexity must match dataset complexity to achieve good generalization.

---

# Phase 8 — Early Stopping

Training was modified to stop when validation loss stops improving.

### Implementation

- track best validation loss
- stop training after `patience` epochs without improvement

Best model weights are saved using:

```python
model.state_dict()
```

### Key Insight

Early stopping acts as an **implicit regularizer** that prevents overtraining.

---

# Phase 9 — Learning Rate Scheduling

Learning rate scheduling improves training stability.

Implemented using:

```python
torch.optim.lr_scheduler.StepLR
```

Example schedule:

```
Epoch 0   lr = 0.005
Epoch 50  lr = 0.0025
Epoch 100 lr = 0.00125
```

### Key Insight

Large learning rates help **rapid early learning**, while smaller learning rates enable **fine tuning near minima**.

---

# Phase 10 — Optimizer Comparison

Optimizers compared:

- SGD
- SGD + Momentum
- Adam

### Observation

Adam converges significantly faster because it uses:

- adaptive learning rates
- momentum-based gradient estimation

### Key Insight

Optimizer choice significantly impacts **training speed and convergence behavior**.

---

# Phase 11 — Weight Initialization

Different initialization strategies were tested:

- very small weights
- very large weights
- He initialization

### Observations

| Initialization | Behavior |
|---|---|
| Small weights | Vanishing gradients |
| Large weights | Exploding gradients |
| He initialization | Stable training |

### Key Insight

Proper initialization maintains **stable signal propagation through deep networks**.

---

# Phase 12 — Batch Normalization

BatchNorm normalizes layer activations:

$$
x̂ = (x - μ) / sqrt(σ² + ε)
$$

Then applies learnable parameters:

$$
y = γx̂ + β
$$

### Benefits

- stabilizes gradient flow
- allows higher learning rates
- accelerates convergence

### Observation

BatchNorm is most beneficial in **deep networks**, while small regression problems may not require it.

---

# Phase 13 — Softmax + CrossEntropy (Classification)

The project transitions from **regression to classification**.

### Softmax

Converts logits into probabilities:

$$
σ(z_i) = e^{z_i} / Σ e^{z_j}
$$

### CrossEntropy Loss

Measures prediction error:

$$
L = - Σ y_i log(p_i)
$$

PyTorch provides:

```python
nn.CrossEntropyLoss()
```

which combines **Softmax + Log + CrossEntropy** in a numerically stable implementation.

---

# Classification Experiment

A synthetic circular dataset was generated using:

```
x² + y² > 1
```

The neural network successfully learned the **nonlinear circular decision boundary** separating the two classes.

---

# Repository Structure

```
.
├── linear_regression_scratch.py
├── polynomial_regression.py
├── neural_network_basic.py
├── nn_train_test_split.py
├── nn_l2_dropout.py
├── nn_minibatch_dataloader.py
├── nn_minibatch_early_stopping.py
├── nn_lr_scheduler.py
├── optimizer_comparison.py
├── weight_initialization_experiments.ipynb
├── batch_normalization_experiment.ipynb
├── softmax_crossentropy_classification.ipynb
└── README.md
```

---

# Skills Developed

This repository demonstrates practical understanding of:

- gradient descent optimization
- neural network architecture design
- feature scaling
- mini-batch training pipelines
- overfitting diagnostics
- regularization techniques
- optimizer dynamics
- learning rate scheduling
- gradient stability
- classification models

---

# Future Work

Upcoming topics:

- decision boundary visualization
- convolutional neural networks (CNNs)
- residual connections
- recurrent neural networks (RNNs)
- attention mechanisms
- transformer architectures

---

# Goal of This Repository

The long-term goal is to build deep learning models **from first principles**, understand training dynamics deeply, and develop the ability to **read and implement modern research papers**.
