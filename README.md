# VRLS_main

This repository provides an example implementation for the paper:  
**"Addressing Label Shift in Distributed Learning via Entropy Regularization"**  
📄 [Read on arXiv](https://arxiv.org/abs/2502.02544)

## 📌 Overview
This repository contains implementations for:
1. **Density Ratio Estimation**: Fine-tuning a ResNet-18 model on Tiny-ImageNet using standard cross-entropy or entropy regularization, followed by convex optimization to estimate the label distribution on test sets.
2. **Training a Global Model with Importance Weighting** under a multi-node setting.

The recommended environment for running these experiments is **Google Colab**.

---

## 🚀 Getting Started

### 📂 1. Density Ratio Estimation
This experiment involves fine-tuning a **ResNet-18** model on **Tiny-ImageNet** with:
- **Standard Cross-Entropy Loss**
- **Entropy Regularization** (to encourage uniform prediction confidence)
- **Convex Optimization** to estimate the label distribution on test datasets

#### 🔧 **Run in Google Colab**
Execute the following notebook:
```bash
Density_Ratio_Estimation.ipynb
```

## 🔥 What This Will Do

By running the above notebook, you will:

- 🏋 **Train ResNet-18** on **Tiny-ImageNet**
- 🔄 **Apply Entropy Regularization** to encourage uniform prediction confidence
- 📊 **Perform Convex Optimization** to estimate label distributions on test datasets

---

## 🌍 2. Train a Global Model with Importance Weighting

This part demonstrates **Importance Weighting** under a multi-node federated learning setting.

### ⚡ **Run an example training setup**
To train a global model on **Fashion-MNIST (FMNIST)** under a **5-client label shift setting**, execute:
```bash
python runner_target_shift.py --dataset fmnist --shift 5clients --batch-size 64 --num-steps 500 --client-mode multi
```
### 🏗 What This Command Does
- 🖥 **Trains a global model** using **importance weighting**
- 📡 Simulates a **5-client federated learning setup** under label shift
- ⚙ Uses a **batch size of 64** and runs for **500 training steps**
- 🏎 Supports **CPU execution** (can be adapted for GPU)

---

## 🛠 Dependencies

To ensure proper execution, install the required dependencies:
```bash
pip install torch torchvision tqdm cvxpy numpy
```

📧 Contact

For any questions or discussions, feel free to open an issue or reach out!

