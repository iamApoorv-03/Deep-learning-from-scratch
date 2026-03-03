#  Deep Learning Foundations: From Scratch to Mini-Batch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Status](https://img.shields.io/badge/Status-Active_Development-green?style=for-the-badge)

##  Overview
This repository documents a structured journey of building **Deep Learning foundations from scratch** using PyTorch. Instead of treating models as black boxes, the goal is to:

* **Understand Optimization:** Deconstruct the mathematics of gradients.
* **Build Step-by-Step:** Move from raw tensors to high-level APIs.
* **Scientific Experimentation:** Analyze noise, bias–variance tradeoffs, and overfitting.
* **Professional Pipelines:** Implement industry-standard training loops.

---

##  Evolution Phases

<details open>
<summary><b>Phase 1 — Linear Regression From Scratch</b></summary>

* **Concepts:** Gradient Descent (GD), Learning rate behavior, Mean Squared Error (MSE).
* **Mechanics:** `requires_grad=True`, `loss.backward()`, manual parameter updates, and gradient zeroing.
* **Key Insight:** Optimization stability is highly dependent on feature scaling; high learning rates lead to divergence (overshooting).
</details>

<details>
<summary><b>Phase 2 — Noise & Bias–Variance Tradeoff</b></summary>

* **Experiments:** Introduced Gaussian noise to datasets to observe parameter stability.
* **Key Insight:** Increasing noise increases irreducible error. Overfitting occurs when the model begins to "memorize" this noise rather than the underlying signal.
</details>

<details>
<summary><b>Phase 3 — Polynomial Regression</b></summary>

* **Experiments:** Added $x^2$ features; implemented feature normalization.
* **Key Insight:** Feature scaling is critical. Without normalization, gradients explode because the optimization geometry becomes too "stretched."
</details>

<details>
<summary><b>Phase 4 — Neural Networks</b></summary>

* **Architecture:** `nn.Sequential` with ReLU non-linearity.
* **Mechanics:** `model.train()` vs `model.eval()`, and `torch.no_grad()` for validation.
* **Key Insight:** ReLU introduces piecewise linear behavior, allowing the model to fit non-linear patterns.
</details>

<details>
<summary><b>Phase 5 — Mini-Batch Training (DataLoader)</b></summary>

* **Implementation:** `TensorDataset` and `DataLoader`.
* **Key Insight:** Batch training introduces "gradient noise" which can actually help in escaping local minima. Average loss per epoch is a more stable metric than total loss.
</details>

<details>
<summary><b>Phase 6 — Regularization (L2 & Dropout)</b></summary>

* **Techniques:** Weight Decay (L2) and Dropout layers.
* **Key Insight:** Regularization is a tool to trade bias for variance. It is only useful when model capacity significantly exceeds data complexity.
</details>

<details>
<summary><b>Phase 7 — Overfitting Diagnosis</b></summary>

* **Experiments:** Small data samples vs. high-capacity (512-512) architectures.
* **Observation:** Evaluation loss divergence is the primary indicator of generalization failure.
</details>

<details open>
<summary><b>Phase 8 — Professional Enhancements (Current)</b></summary>

* **Features:** Early stopping logic, learning curve plotting, and modular train/eval separation.
* **Goal:** Transforming toy scripts into production-style pipelines.
</details>

---

##  Repository Structure
```text
.
├── linear_regression_scratch.py  # Manual GD & Autograd
├── polynomial_regression.py      # Feature engineering & scaling
├── neural_network_basic.py       # Non-linear activations
├── nn_train_test_split.py        # Validation strategies
├── nn_l2_dropout.py              # Regularization experiments
├── nn_minibatch_dataloader.py    # Efficient data pipelines
└── README.md
