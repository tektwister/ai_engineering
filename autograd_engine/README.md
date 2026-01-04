# Autograd Engine

A lightweight autograd engine implemented in Go, inspired by [Andre Karpathy's Micrograd](https://github.com/karpathy/micrograd).

This engine implements a scalar-valued autograd engine and a neural network library on top of it. It implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library with a PyTorch-like API.

## Features

- Scalar-valued Autograd engine (`Value` struct)
- Operations: Add, Sub, Mul, Pow, Tanh, ReLU
- Automatic backward pass using topological sort
- Neural Network building blocks: `Neuron`, `Layer`, `MLP`
- No external dependencies (uses standard library `math` and `math/rand`)

## Usage

### Basic Autograd

```go
package main

import (
    "fmt"
    "github.com/tektwister/ai_engineering/autograd_engine"
)

func main() {
    a := autograd.NewValue(2.0)
    b := autograd.NewValue(-3.0)
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
go run cmd/main.go
```
