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
* **Key Insight:** Batch training introduces stochasticity into gradient updates ("gradient noise"), which can help optimization escape poor local minima. Tracking **average loss per epoch** provides more stable monitoring than raw batch losses.
</details>

<details>
<summary><b>Phase 6 — Regularization (L2 & Dropout)</b></summary>

* **Techniques:** Weight Decay (L2 Regularization) and Dropout layers.
* **Key Insight:** Regularization reduces variance but increases bias. It is beneficial when **model capacity is larger than dataset complexity**, helping prevent memorization of noise.
</details>

<details>
<summary><b>Phase 7 — Overfitting Diagnosis</b></summary>

* **Experiments:** Reduced dataset size and increased network capacity (e.g., 512-512 hidden layers).
* **Observation:** Training loss continues decreasing while evaluation loss increases — a classic sign of **overfitting**.
* **Insight:** Model capacity must be balanced with data complexity to ensure good generalization.
</details>

<details>
<summary><b>Phase 8 — Early Stopping (Training Control)</b></summary>

* **Implementation:** Monitoring validation loss during training and stopping when improvement stagnates.
* **Mechanism:**  
  - Track the **best validation loss** observed so far.  
  - If validation loss does not improve for a defined number of epochs (`patience`), training stops early.
* **Checkpointing:** The model parameters corresponding to the **best validation performance** are stored using `state_dict()` and restored after stopping.
* **Key Insight:** Early stopping acts as an **implicit regularizer**, preventing unnecessary training once generalization stops improving.
</details>

<details open>
<summary><b>Phase 9 — Training Diagnostics & Learning Curves (Current)</b></summary>

* **Features Implemented:**
  - Training vs validation loss tracking
  - Learning curve visualization
  - Structured training/evaluation loops
* **Goal:** Develop the ability to **diagnose training dynamics** such as convergence behavior, underfitting, and overfitting using visual analysis.
</details>

---
## Phase 10 — Learning Rate Scheduling 

Learning rate scheduling improves training stability.

Implemented using:
torch.optim.lr_scheduler.StepLR


Purpose:

- Large steps during early training
- Small steps during fine-tuning

Learning rate decay works together with **early stopping** to produce stable convergence.

---


##  Repository Structure
```text
.
├── linear_regression_scratch.py   # Manual GD & Autograd
├── polynomial_regression.py       # Feature engineering & scaling
├── neural_network_basic.py          # Non-linear activations
├── nn_train_test_split.py            # Validation strategies
├── nn_l2_dropout.py                   # Regularization experiments
├── nn_minibatch_dataloader.py         # Efficient data pipelines
├── nn_minibatch_early_stopping.py     # Early stopping & best model checkpointing
├── nn_lr_scheduler.py                 # Learning rate scheduling for adaptive training convergence
└── README.md            

