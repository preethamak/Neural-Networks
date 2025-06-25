
# 🧠 NEURAL-NETWORKS

**Unlock Intelligent Insights Through Seamless Learning**  
_A step-by-step implementation of a neural network and autodiff engine from scratch in Python._

![Last Commit](https://img.shields.io/github/last-commit/preethamak/Neural-Networks?color=blue)  
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-informational)  

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How it Works](#how-it-works)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Visualization](#visualization)
- [License](#license)

---

## 📌 Overview

This project builds a simple neural network and an automatic differentiation engine (like [micrograd](https://github.com/karpathy/micrograd)) from scratch using Python. It's made to deeply understand how neural networks learn via backpropagation.

---

## ✨ Features

- 🔢 **Scalar Value Class** for gradient tracking  
- 🔄 **Automatic Differentiation** via backpropagation  
- ➕ Overloaded Operators: `+`, `*`, `**`, `-`, `/`  
- 📈 **Activation Functions**: `tanh`, `exp`  
- 🧱 Components:
  - `Neuron`
  - `Layer`
  - `MLP (Multi-Layer Perceptron)`
- 📊 **Training Loop** using gradient descent  
- 🧠 **Computation Graph Visualization** via `graphviz`  

---

## ⚙️ How it Works

The engine revolves around the `Value` class:
- Tracks `data`, `grad`, `_op`, `_prev` values
- Constructs a **computation graph**
- Implements a `backward()` method to compute gradients using the chain rule
- Supports topological sorting for proper backprop order

---

## 📦 Dependencies

Ensure Python 3 and the following libraries are installed:

```bash
pip install numpy matplotlib graphviz torch
```

> 🔧 You may need to install the Graphviz **software** separately from the Python package.

---

## 🚀 Usage

1. Clone the repo:
    ```bash
    git clone https://github.com/preethamak/Neural-Networks.git
    cd Neural-Networks
    ```

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Neural Network.ipynb"
    ```

3. Run the notebook cells and follow the guided steps.

---

## 🧩 Visualization

The `graphviz` integration renders computation graphs to help you understand how each node connects and how gradients flow during backpropagation.

---

## 📄 License

This project is licensed under the MIT License.
