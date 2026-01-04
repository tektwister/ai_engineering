package matmul

import (
	"fmt"
	"runtime"
	"sync"
)

// DefaultBlockSize is the default block size for tiled matrix multiplication.
// Chosen to fit in L1 cache (typically 32KB). For 64-bit floats:
// 3 blocks of 64x64 ≈ 24KB which fits well in L1 cache.
const DefaultBlockSize = 64

// Kernel represents a matrix multiplication implementation.
type Kernel int

const (
	// Naive - Simple O(n³) algorithm with i,j,k loop order.
	Naive Kernel = iota
	// NaiveIKJ - O(n³) with i,k,j loop order (better cache locality for row-major).
	NaiveIKJ
	// Blocked - Tiled/blocked algorithm for better cache utilization.
	Blocked
	// BlockedParallel - Parallel tiled algorithm using goroutines.
	BlockedParallel
	// TransposedB - Pre-transpose B for better memory access pattern.
	TransposedB
	// SIMD - Placeholder for SIMD-optimized version (requires assembly).
	SIMD
)

// String returns the name of the kernel.
func (k Kernel) String() string {
	names := []string{"Naive", "NaiveIKJ", "Blocked", "BlockedParallel", "TransposedB", "SIMD"}
	if int(k) < len(names) {
		return names[k]
	}
	return "Unknown"
}

// Multiply performs matrix multiplication C = A × B using the specified kernel.
func Multiply(a, b *Matrix, kernel Kernel) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, fmt.Errorf("incompatible dimensions: A(%dx%d) × B(%dx%d)", a.Rows, a.Cols, b.Rows, b.Cols)
	}

	switch kernel {
	case Naive:
		return multiplyNaive(a, b), nil
	case NaiveIKJ:
		return multiplyNaiveIKJ(a, b), nil
	case Blocked:
		return multiplyBlocked(a, b, DefaultBlockSize), nil
	case BlockedParallel:
		return multiplyBlockedParallel(a, b, DefaultBlockSize), nil
	case TransposedB:
		return multiplyTransposedB(a, b), nil
	case SIMD:
		// Fall back to blocked parallel for now
		return multiplyBlockedParallel(a, b, DefaultBlockSize), nil
	default:
		return nil, fmt.Errorf("unknown kernel: %d", kernel)
	}
}

// MatMul is a convenience function that uses the best available kernel.
func MatMul(a, b *Matrix) (*Matrix, error) {
	return Multiply(a, b, BlockedParallel)
}

// multiplyNaive implements the standard O(n³) matrix multiplication.
// Loop order: i, j, k
// This is the most straightforward but often slowest due to poor cache locality.
func multiplyNaive(a, b *Matrix) *Matrix {
	m, n, k := a.Rows, b.Cols, a.Cols
	c := NewMatrix(m, n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a.At(i, l) * b.At(l, j)
			}
			c.Set(i, j, sum)
		}
	}
	return c
}

// multiplyNaiveIKJ uses i,k,j loop ordering for better cache locality.
// In row-major storage, accessing b.At(k, j) with varying j utilizes
// spatial locality since consecutive j values are adjacent in memory.
func multiplyNaiveIKJ(a, b *Matrix) *Matrix {
	m, n, k := a.Rows, b.Cols, a.Cols
	c := NewMatrix(m, n)

	for i := 0; i < m; i++ {
		for l := 0; l < k; l++ {
			aik := a.At(i, l)
			for j := 0; j < n; j++ {
				c.Data[i*n+j] += aik * b.Data[l*n+j]
			}
		}
	}
	return c
}

// multiplyBlocked implements tiled/blocked matrix multiplication.
// The matrix is divided into blocks that fit in cache, reducing cache misses.
//
// For matrices A (M×K) and B (K×N):
// - Divide into blocks of size blockSize × blockSize
// - Multiply block by block
//
// This achieves O(n³/(B*√M)) cache misses where B is block size and M is cache size.
func multiplyBlocked(a, b *Matrix, blockSize int) *Matrix {
	m, n, k := a.Rows, b.Cols, a.Cols
	c := NewMatrix(m, n)

	// Iterate over blocks
	for i0 := 0; i0 < m; i0 += blockSize {
		iEnd := min(i0+blockSize, m)
		for j0 := 0; j0 < n; j0 += blockSize {
			jEnd := min(j0+blockSize, n)
			for k0 := 0; k0 < k; k0 += blockSize {
				kEnd := min(k0+blockSize, k)

				// Multiply block (i0:iEnd, k0:kEnd) × (k0:kEnd, j0:jEnd)
				// and add to (i0:iEnd, j0:jEnd)
				for i := i0; i < iEnd; i++ {
					for l := k0; l < kEnd; l++ {
						aik := a.At(i, l)
						for j := j0; j < jEnd; j++ {
							c.Data[i*n+j] += aik * b.Data[l*n+j]
						}
					}
				}
			}
		}
	}
	return c
}

// multiplyBlockedParallel is a parallel version of blocked matrix multiplication.
// Each row of blocks is processed by a separate goroutine.
func multiplyBlockedParallel(a, b *Matrix, blockSize int) *Matrix {
	m, n, k := a.Rows, b.Cols, a.Cols
	c := NewMatrix(m, n)

	// Number of worker goroutines
	numWorkers := runtime.NumCPU()
	if numWorkers > m {
		numWorkers = m
	}

	// Each worker handles a range of row blocks
	rowsPerWorker := (m + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			startRow := workerID * rowsPerWorker
			endRow := min(startRow+rowsPerWorker, m)
			if startRow >= m {
				return
			}

			// Process blocks assigned to this worker
			for i0 := startRow; i0 < endRow; i0 += blockSize {
				iEnd := min(i0+blockSize, endRow)
				for j0 := 0; j0 < n; j0 += blockSize {
					jEnd := min(j0+blockSize, n)
					for k0 := 0; k0 < k; k0 += blockSize {
						kEnd := min(k0+blockSize, k)

						// Multiply block
						for i := i0; i < iEnd; i++ {
							for l := k0; l < kEnd; l++ {
								aik := a.At(i, l)
								for j := j0; j < jEnd; j++ {
									c.Data[i*n+j] += aik * b.Data[l*n+j]
								}
							}
						}
					}
				}
			}
		}(w)
	}

	wg.Wait()
	return c
}

// multiplyTransposedB pre-transposes matrix B for better memory access.
// When B is transposed, both A and B^T can be accessed row-by-row,
// maximizing cache line utilization.
func multiplyTransposedB(a, b *Matrix) *Matrix {
	m, n, k := a.Rows, b.Cols, a.Cols
	c := NewMatrix(m, n)

	// Pre-transpose B
	bT := b.Transpose()

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			// Now both rows of A and rows of B^T are accessed sequentially
			aRowBase := i * k
			bTRowBase := j * k
			for l := 0; l < k; l++ {
				sum += a.Data[aRowBase+l] * bT.Data[bTRowBase+l]
			}
			c.Set(i, j, sum)
		}
	}
	return c
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
