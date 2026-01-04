# Matrix Multiplication Kernel

A comprehensive Go implementation of matrix multiplication kernels showcasing various optimization techniques from naive O(n³) to cache-optimized blocked and parallel versions.

## Overview

This module provides multiple matrix multiplication algorithms designed to demonstrate the impact of:
- **Memory access patterns** (cache locality)
- **Loop ordering** (i,j,k vs i,k,j)
- **Blocking/tiling** (fitting data in L1/L2 cache)
- **Parallelization** (multi-core utilization)
- **Algorithm optimization** (Strassen's algorithm)

## Features

### Matrix Multiplication Kernels

| Kernel | Description | Complexity | Use Case |
|--------|-------------|------------|----------|
| `Naive` | Standard i,j,k loop order | O(n³) | Baseline |
| `NaiveIKJ` | Optimized i,k,j loop order | O(n³) | Better cache locality |
| `Blocked` | Cache-blocked/tiled | O(n³) | L1/L2 cache friendly |
| `BlockedParallel` | Parallel tiled | O(n³/p) | Multi-core systems |
| `TransposedB` | Pre-transpose B matrix | O(n³) | Vectorization friendly |
| `Strassen` | Divide and conquer | O(n^2.807) | Very large matrices |

### Additional Operations

- **Element-wise**: Add, Sub, Hadamard (element-wise multiply), Scale
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **Reductions**: Sum, Mean, Max, Min, Norm
- **Utilities**: Transpose, Clone, Equal (with tolerance)
- **Random**: Randn, Rand, RandRange

## Installation

```bash
cd matrix_multiplication_kernel
go mod tidy
```

## Usage

### Basic Matrix Multiplication

```go
package main

import (
    "fmt"
    matmul "github.com/tektwister/ai_engineering/matrix_multiplication_kernel"
)

func main() {
    // Create matrices
    a := matmul.NewMatrixFromSlice([][]float64{
        {1, 2, 3},
        {4, 5, 6},
    })
    b := matmul.NewMatrixFromSlice([][]float64{
        {7, 8},
        {9, 10},
        {11, 12},
    })

    // Multiply using the best kernel (BlockedParallel)
    c, err := matmul.MatMul(a, b)
    if err != nil {
        panic(err)
    }
    fmt.Println(c)
}
```

### Choosing a Specific Kernel

```go
// Use a specific kernel
result, _ := matmul.Multiply(a, b, matmul.Blocked)

// Use Strassen's algorithm for very large matrices
result, _ = matmul.Strassen(a, b)
```

### Benchmarking

```go
// Run comprehensive benchmark
config := matmul.DefaultBenchmarkConfig()
config.Sizes = []int{128, 256, 512, 1024}
results := matmul.Benchmark(config)
matmul.PrintBenchmarkResults(results)

// Compare two kernels
base, opt, speedup := matmul.CompareKernels(512, matmul.Naive, matmul.BlockedParallel)
fmt.Printf("Speedup: %.2fx\n", speedup)
```

## Running Tests

```bash
go test -v ./...
```

## Running Benchmarks

```bash
# Run Go benchmarks
go test -bench=. -benchmem

# Use the example CLI
go run ./cmd/benchmark/main.go
```

## Algorithm Details

### Loop Order Optimization

The standard i,j,k loop order causes poor cache utilization because accessing `B[k][j]` with varying `k` (inner loop) results in strided memory access:

```
// Naive (bad cache locality for B)
for i := 0; i < M; i++ {
    for j := 0; j < N; j++ {
        for k := 0; k < K; k++ {
            C[i][j] += A[i][k] * B[k][j]  // B accessed by column
        }
    }
}
```

The i,k,j order allows sequential access for both B rows and C rows:

```
// IKJ (good cache locality)
for i := 0; i < M; i++ {
    for k := 0; k < K; k++ {
        for j := 0; j < N; j++ {
            C[i][j] += A[i][k] * B[k][j]  // B row accessed sequentially
        }
    }
}
```

### Blocked/Tiled Multiplication

Blocking divides matrices into tiles that fit in cache, dramatically reducing cache misses:

```
// Blocked algorithm pseudo-code
for i_block := 0 to M step BLOCK_SIZE:
    for j_block := 0 to N step BLOCK_SIZE:
        for k_block := 0 to K step BLOCK_SIZE:
            // Multiply blocks - data fits in L1 cache
            for i := i_block to i_block + BLOCK_SIZE:
                for k := k_block to k_block + BLOCK_SIZE:
                    for j := j_block to j_block + BLOCK_SIZE:
                        C[i][j] += A[i][k] * B[k][j]
```

The default block size (64) is chosen to fit three 64×64 blocks of float64 (~24KB) in typical L1 cache (~32KB).

### Strassen's Algorithm

Uses divide-and-conquer with 7 multiplications instead of 8 for 2×2 block matrices:

- Standard: 8 multiplications, O(n³)
- Strassen: 7 multiplications, O(n^2.807)

Speedup becomes significant for n > 500-1000.

## Performance Expectations

On a typical modern CPU (Intel Core i7, AMD Ryzen):

| Size | Naive | NaiveIKJ | Blocked | BlockedParallel |
|------|-------|----------|---------|-----------------|
| 256 | ~1 GFLOPS | ~3 GFLOPS | ~5 GFLOPS | ~15 GFLOPS |
| 512 | ~0.8 GFLOPS | ~2.5 GFLOPS | ~4 GFLOPS | ~12 GFLOPS |
| 1024 | ~0.5 GFLOPS | ~2 GFLOPS | ~3.5 GFLOPS | ~10 GFLOPS |

*Note: Actual performance varies based on CPU, memory, and system load.*

## Future Enhancements

- [ ] SIMD optimizations (AVX2/AVX512)
- [ ] GPU acceleration (CUDA/OpenCL bindings)
- [ ] Sparse matrix support
- [ ] Mixed precision (FP16/BF16)
- [ ] Memory pooling for reduced allocations
- [ ] Out-of-core computation for matrices larger than RAM

## References

1. [Strassen Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Strassen_algorithm)
2. [Tiling/Blocking for Matrix Multiplication](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf)
3. [Fast Matrix Multiplication - MIT OpenCourseWare](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/)
