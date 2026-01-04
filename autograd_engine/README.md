# Autograd Engine (Micrograd Implementation)

A lightweight, educational autograd engine implemented in Go, inspired by [Andre Karpathy's Micrograd](https://github.com/karpathy/micrograd). This module implements automatic differentiation (backpropagation) from scratch and a simple neural network library with a PyTorch-like API.

## About Autograd

**Automatic Differentiation (Autodiff)** is the foundational technique that powers modern deep learning. Instead of manually computing gradients or using numerical approximation, autodiff automatically computes exact gradients using the chain rule.

### Why Autograd?

1. **Backpropagation**: Core algorithm for training neural networks via gradient descent.
2. **Chain Rule**: Automatically applies mathematical differentiation rules.
3. **Exact Gradients**: Avoids numerical instability (unlike finite differences).
4. **Composability**: Build complex compute graphs from simple operations.

### Applications

- Training neural networks (MLPs, CNNs, RNNs, Transformers)
- Physics simulations
- Optimization problems
- Scientific computing

## How Autograd Works

### Computational Graphs

A computational graph represents the sequence of operations that compute the output from inputs.

```
Example: f(a, b) = a*b + b
Graph:
    a ──┐
        │ [Mul] ──┐
    b ──┤         │ [Add] ──> c
        └─────────┤
             b ──┘

Nodes: a, b (inputs), *, +, c (output)
Edges: Data flow and dependency relationships
```

### Forward Pass

Compute the value by following the graph:

```
a = 2, b = 3
a * b = 6
6 + b = 9
```

### Backward Pass (Backpropagation)

Compute gradients by reversing the graph using the chain rule:

```
dc/dc = 1
dc/d(a*b) = 1
dc/db (from add) = 1
dc/da = 1 * 3 = 3
dc/db (from mul) = 2 * 1 = 2
dc/db (total) = 1 + 2 = 3
```

## Project Structure

```
autograd_engine/
├── engine.go           # Value struct and operations (Tanh, ReLU, Add, Mul, etc.)
├── nn.go               # Neural network components (Neuron, Layer, MLP)
├── cmd/
│   └── main.go         # Training example: binary classification task
├── go.mod              # Go module definition
└── README.md           # This file
```

## Architecture

### Core Components

#### 1. **Value** (engine.go)

Represents a scalar value with automatic differentiation support.

```go
type Value struct {
    data     float64      // The actual numerical value
    grad     float64      // Accumulated gradient
    prev     []*Value     // Children nodes (dependencies)
    op       string       // Operation that created this node
    backward func()       // Function to propagate gradients
}
```

**Key Methods:**
- `Add(other)` → addition
- `Sub(other)` → subtraction
- `Mul(other)` → multiplication
- `Pow(exponent)` → exponentiation
- `Tanh()` → tanh activation
- `ReLU()` → ReLU activation
- `Backward()` → compute all gradients

#### 2. **Neuron** (nn.go)

A single artificial neuron: `y = activation(w·x + b)`

```go
type Neuron struct {
    w   []*Value         // Weights
    b   *Value           // Bias
    act bool             // Non-linearity (ReLU)
}
```

**Key Method:**
- `Call(x)` → forward pass through the neuron

#### 3. **Layer** (nn.go)

A collection of neurons with the same input size.

```go
type Layer struct {
    neurons []*Neuron
}
```

**Key Method:**
- `Call(x)` → output from each neuron in parallel

#### 4. **MLP** (nn.go)

Multi-Layer Perceptron: stacks layers into a neural network.

```go
type MLP struct {
    layers []*Layer
}
```

**Constructor:**
```go
// Example: 3 inputs → 4 hidden neurons → 4 hidden neurons → 1 output
model := NewMLP(3, []int{4, 4, 1})
```

**Key Method:**
- `Call(x)` → forward pass through all layers

## Features

- Scalar-valued Autograd engine (`Value` struct)
- Operations: Add, Sub, Mul, Pow, Tanh, ReLU
- Automatic backward pass using topological sort
- Neural Network building blocks: `Neuron`, `Layer`, `MLP`
- No external dependencies (uses standard library `math` and `math/rand`)

    c := autograd.NewValue(10.0)
    
    // e = a*b + c
    d := a.Mul(b)
    e := d.Add(c)
    
    fmt.Println(e.Data()) // 4.0
    
    // Compute gradients
    e.Backward()
    
    fmt.Println(a.Grad()) // -3.0 (de/da = b)
    fmt.Println(b.Grad()) // 2.0 (de/db = a)
}
```

### Neural Network Example

See `cmd/main.go` for a complete training loop example.

```go
// Create a multi-layer perceptron
// Inputs: 3
// Hidden Layers: 4, 4
// Output: 1
model := autograd.NewMLP(3, []int{4, 4, 1})

// Forward pass
out := model.Call(inputs)

// Compute loss...

// Backward
model.ZeroGrad()
loss.Backward()

// Update parameters
for _, p := range model.Parameters() {
    p.SetData(p.Data() - learningRate * p.Grad())
}
```

## Running the Example

```bash
cd autograd_engine
go run cmd/main.go
```

**Expected Output:**
```
Starting training...
Step 0: loss 2.574305
Step 1: loss 2.214562
Step 2: loss 1.958234
...
Step 19: loss 0.015234

Final predictions:
Input: 0, Target: 1.000000, Prediction: 0.984521
Input: 1, Target: -1.000000, Prediction: -0.987342
Input: 2, Target: -1.000000, Prediction: -0.976543
Input: 3, Target: 1.000000, Prediction: 0.995231
```

## Computational Complexity

### Forward Pass
- **Time**: O(E) where E = number of operations
- **Space**: O(N) where N = number of Values stored in graph

### Backward Pass
- **Time**: O(E) - traverses graph once in reverse
- **Space**: O(E) - temporary storage for topological sort

### Training
- **Per Iteration**: O(E × iterations)
- **Memory**: O(N) - parameters + intermediate values

## Activation Functions

### ReLU (Rectified Linear Unit)
```
y = max(0, x)
dy/dx = 1 if x > 0, else 0
```
Best for hidden layers due to efficiency and non-linearity.

### Tanh (Hyperbolic Tangent)
```
y = tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
dy/dx = 1 - y^2
```
Outputs in range [-1, 1]; good for regression or normalized inputs.

## Important Concepts

### Gradient Accumulation

Gradients from multiple paths are accumulated:
```go
v.grad += ...  // Note the +=, not =
```

### Topological Sort

Ensures derivatives are computed in the correct order:

```
Compute graph:
    a ── b ── c ── loss

Topological order: [a, b, c, loss]
Backward order: [loss, c, b, a]
```

### Zero Gradients

Before each backward pass, reset gradients to avoid accumulation across iterations:
```go
model.ZeroGrad()
```

## Limitations & Extensions

### Current Limitations

1. **Scalar-only**: Handles individual scalar values; vectors/matrices via loops
2. **Single-threaded**: No parallelization
3. **No optimization**: Uses basic SGD; no Adam, momentum, etc.
4. **Memory**: Stores entire compute graph (not memory-efficient for large networks)

### Possible Extensions

- **Batching**: Process multiple examples efficiently
- **Tensor operations**: Multi-dimensional arrays
- **GPU acceleration**: Compute on accelerators
- **Advanced optimizers**: Adam, RMSprop, etc.
- **Regularization**: L1/L2, dropout
- **Checkpointing**: Save/load trained models

## Performance Notes

- **Random Initialization**: Neuron weights use uniform distribution [-1, 1]
- **Learning Rate**: Typically 0.01-0.1 for small networks
- **Convergence**: May require 20-100 iterations for toy datasets

## Learning Path

This module teaches:

1. **Computation Graphs**: How operations form DAGs
2. **Forward Pass**: Evaluating expressions
3. **Backpropagation**: Computing gradients via chain rule
4. **Neural Network Building**: Composing layers and neurons
5. **Gradient Descent**: Parameter optimization

## Next Steps in the Roadmap

After understanding autograd:

- **Tokenizer**: How to preprocess text for models
- **Transformer**: Build attention mechanisms
- **Larger Networks**: Train deeper models with more data
- **Optimization**: Implement advanced training techniques

## References

- **Micrograd**: [GitHub](https://github.com/karpathy/micrograd) by Andre Karpathy
- **Backpropagation**: [The Backpropagation Algorithm](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- **Automatic Differentiation**: [Autodiff on Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)
- **Neural Networks**: [3Blue1Brown's Neural Networks Playlist](https://www.youtube.com/watch?v=aircAruvnKk)

## License

See the root repository LICENSE file.
