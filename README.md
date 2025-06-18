# Neural Network from Scratch in Python

This Jupyter Notebook provides a step-by-step implementation of a simple neural network and an automatic differentiation engine (similar to Andrej Karpathy's micrograd) from scratch using Python. It's designed to be an educational tool to understand the fundamentals of how neural networks learn through backpropagation.

## Description

The notebook walks through:
1.  Defining basic mathematical functions and manually calculating their derivatives.
2.  Introducing the `Value` class, a core component that encapsulates scalar values and enables automatic gradient computation.
3.  Overloading arithmetic operators (`+`, `*`, `**`) and implementing activation functions (`tanh`, `exp`) for the `Value` class to build and track computation graphs.
4.  Implementing the `backward()` pass to automatically compute gradients for all parameters in a computation graph using the chain rule.
5.  Visualizing these computation graphs using `graphviz` to better understand the forward and backward passes.
6.  Building `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron) classes using the `Value` object.
7.  Demonstrating a simple training loop where the MLP learns to fit a small dataset.

## Features

*   **Scalar `Value` Object:** Tracks operations and gradients.
*   **Automatic Differentiation:** Implements backpropagation for gradient calculation.
*   **Supported Operations:**
    *   Addition (`+`)
    *   Multiplication (`*`)
    *   Power (`**`)
    *   Tanh activation
    *   Exponentiation (`exp`)
    *   Subtraction (`-`), Division (`/`), Negation (`-`) (derived from the above)
*   **Neural Network Components:**
    *   `Neuron` class
    *   `Layer` class (a collection of neurons)
    *   `MLP` class (a multi-layer perceptron)
*   **Computation Graph Visualization:** Uses `graphviz` to draw the network's structure and the flow of data and gradients.
*   **Training Loop:** A basic example of training the MLP using gradient descent.

## How it Works (The `Value` Object and Backpropagation)

The core of the automatic differentiation engine is the `Value` class:
*   Each `Value` object stores a scalar `data` value and its associated `grad` (gradient), initialized to 0.
*   When arithmetic operations or supported functions (like `tanh`) are performed on `Value` objects, a new `Value` object is created. This new object remembers its "children" (the operands or the input `Value`) and the operation (`_op`) that produced it.
*   Crucially, each operation also defines a local `_backward` function. This function knows how to propagate the gradient from the output `Value` back to its input "children" `Value`s based on the chain rule of calculus. For example, for `c = a + b`, `a.grad` will be incremented by `1.0 * c.grad` and `b.grad` will also be incremented by `1.0 * c.grad`.
*   The `backward()` method, when called on a `Value` (typically the final loss value), performs a topological sort of the entire computation graph leading to that `Value`.
*   It then iterates through the sorted nodes in reverse order, calling the `_backward` function for each node. This process accumulates the gradients for all `Value` objects that participated in the computation, effectively performing backpropagation.

## Dependencies

To run this notebook, you'll need Python 3 and the following libraries:

*   `math` (standard library)
*   `numpy`
*   `matplotlib` (for plotting)
*   `graphviz` (for visualizing computation graphs)
    *   You might need to install the Graphviz software separately from the Python library. See Graphviz download page.
*   `torch` (PyTorch is used in one cell for comparison/verification of gradients but is not a core dependency for the custom implementation itself)
*   `random` (standard library, used for weight initialization)

You can typically install the Python libraries using pip:
```bash
pip install numpy matplotlib graphviz torch
