package transformer

import (
	"fmt"
	"math"
	"math/rand"
)

// Tensor represents a multi-dimensional array for neural network computations.
// For simplicity, we use a 1D slice with shape metadata.
type Tensor struct {
	Data  []float64
	Shape []int
}

// NewTensor creates a tensor with the given shape, initialized to zeros.
func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{
		Data:  make([]float64, size),
		Shape: shape,
	}
}

// NewTensorFromData creates a tensor from existing data.
func NewTensorFromData(data []float64, shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if len(data) != size {
		panic(fmt.Sprintf("data length %d does not match shape %v (size %d)", len(data), shape, size))
	}
	return &Tensor{
		Data:  data,
		Shape: shape,
	}
}

// Zeros creates a tensor filled with zeros.
func Zeros(shape ...int) *Tensor {
	return NewTensor(shape...)
}

// Ones creates a tensor filled with ones.
func Ones(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = 1.0
	}
	return t
}

// Randn creates a tensor with random normal values.
func Randn(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64()
	}
	return t
}

// RandnScaled creates a tensor with scaled random normal values (Xavier initialization).
func RandnScaled(scale float64, shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64() * scale
	}
	return t
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return &Tensor{Data: data, Shape: shape}
}

// Size returns the total number of elements.
func (t *Tensor) Size() int {
	return len(t.Data)
}

// Reshape returns a new tensor with the given shape (must have same total size).
func (t *Tensor) Reshape(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if size != len(t.Data) {
		panic(fmt.Sprintf("cannot reshape tensor of size %d to shape %v (size %d)", len(t.Data), shape, size))
	}
	newShape := make([]int, len(shape))
	copy(newShape, shape)
	return &Tensor{Data: t.Data, Shape: newShape}
}

// At gets the value at the specified indices (for 2D or 3D tensors).
func (t *Tensor) At(indices ...int) float64 {
	idx := t.flatIndex(indices...)
	return t.Data[idx]
}

// Set sets the value at the specified indices.
func (t *Tensor) Set(value float64, indices ...int) {
	idx := t.flatIndex(indices...)
	t.Data[idx] = value
}

// flatIndex converts multi-dimensional indices to a flat index.
func (t *Tensor) flatIndex(indices ...int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.Shape), len(indices)))
	}
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	return idx
}

// Add performs element-wise addition.
func (t *Tensor) Add(other *Tensor) *Tensor {
	if len(t.Data) != len(other.Data) {
		panic("tensor size mismatch for Add")
	}
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] += other.Data[i]
	}
	return result
}

// AddScalar adds a scalar to all elements.
func (t *Tensor) AddScalar(s float64) *Tensor {
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] += s
	}
	return result
}

// Sub performs element-wise subtraction.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if len(t.Data) != len(other.Data) {
		panic("tensor size mismatch for Sub")
	}
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] -= other.Data[i]
	}
	return result
}

// Mul performs element-wise multiplication.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if len(t.Data) != len(other.Data) {
		panic("tensor size mismatch for Mul")
	}
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] *= other.Data[i]
	}
	return result
}

// MulScalar multiplies all elements by a scalar.
func (t *Tensor) MulScalar(s float64) *Tensor {
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] *= s
	}
	return result
}

// Div performs element-wise division.
func (t *Tensor) Div(other *Tensor) *Tensor {
	if len(t.Data) != len(other.Data) {
		panic("tensor size mismatch for Div")
	}
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] /= other.Data[i]
	}
	return result
}

// DivScalar divides all elements by a scalar.
func (t *Tensor) DivScalar(s float64) *Tensor {
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] /= s
	}
	return result
}

// MatMul performs matrix multiplication for 2D tensors.
// t: (M, K), other: (K, N) -> result: (M, N)
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	if len(t.Shape) != 2 || len(other.Shape) != 2 {
		panic("MatMul requires 2D tensors")
	}
	m, k1 := t.Shape[0], t.Shape[1]
	k2, n := other.Shape[0], other.Shape[1]
	if k1 != k2 {
		panic(fmt.Sprintf("MatMul dimension mismatch: (%d,%d) x (%d,%d)", m, k1, k2, n))
	}

	result := NewTensor(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < k1; k++ {
				sum += t.At(i, k) * other.At(k, j)
			}
			result.Set(sum, i, j)
		}
	}
	return result
}

// BatchMatMul performs batched matrix multiplication for 3D tensors.
// t: (B, M, K), other: (B, K, N) -> result: (B, M, N)
func (t *Tensor) BatchMatMul(other *Tensor) *Tensor {
	if len(t.Shape) != 3 || len(other.Shape) != 3 {
		panic("BatchMatMul requires 3D tensors")
	}
	b1, m, k1 := t.Shape[0], t.Shape[1], t.Shape[2]
	b2, k2, n := other.Shape[0], other.Shape[1], other.Shape[2]
	if b1 != b2 || k1 != k2 {
		panic(fmt.Sprintf("BatchMatMul dimension mismatch: (%d,%d,%d) x (%d,%d,%d)", b1, m, k1, b2, k2, n))
	}

	result := NewTensor(b1, m, n)
	for b := 0; b < b1; b++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := 0.0
				for k := 0; k < k1; k++ {
					sum += t.At(b, i, k) * other.At(b, k, j)
				}
				result.Set(sum, b, i, j)
			}
		}
	}
	return result
}

// Transpose transposes a 2D tensor.
func (t *Tensor) Transpose() *Tensor {
	if len(t.Shape) != 2 {
		panic("Transpose requires 2D tensor")
	}
	m, n := t.Shape[0], t.Shape[1]
	result := NewTensor(n, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result.Set(t.At(i, j), j, i)
		}
	}
	return result
}

// TransposeLast2D transposes the last two dimensions of a 3D tensor.
// (B, M, N) -> (B, N, M)
func (t *Tensor) TransposeLast2D() *Tensor {
	if len(t.Shape) != 3 {
		panic("TransposeLast2D requires 3D tensor")
	}
	b, m, n := t.Shape[0], t.Shape[1], t.Shape[2]
	result := NewTensor(b, n, m)
	for bi := 0; bi < b; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				result.Set(t.At(bi, i, j), bi, j, i)
			}
		}
	}
	return result
}

// Softmax applies softmax along the last dimension.
func (t *Tensor) Softmax() *Tensor {
	result := t.Clone()

	if len(t.Shape) == 2 {
		// 2D: softmax along last dimension (rows)
		rows, cols := t.Shape[0], t.Shape[1]
		for i := 0; i < rows; i++ {
			// Find max for numerical stability
			maxVal := math.Inf(-1)
			for j := 0; j < cols; j++ {
				if t.At(i, j) > maxVal {
					maxVal = t.At(i, j)
				}
			}
			// Compute exp and sum
			sum := 0.0
			for j := 0; j < cols; j++ {
				result.Set(math.Exp(t.At(i, j)-maxVal), i, j)
				sum += result.At(i, j)
			}
			// Normalize
			for j := 0; j < cols; j++ {
				result.Set(result.At(i, j)/sum, i, j)
			}
		}
	} else if len(t.Shape) == 3 {
		// 3D: softmax along last dimension
		b, m, n := t.Shape[0], t.Shape[1], t.Shape[2]
		for bi := 0; bi < b; bi++ {
			for i := 0; i < m; i++ {
				// Find max for numerical stability
				maxVal := math.Inf(-1)
				for j := 0; j < n; j++ {
					if t.At(bi, i, j) > maxVal {
						maxVal = t.At(bi, i, j)
					}
				}
				// Compute exp and sum
				sum := 0.0
				for j := 0; j < n; j++ {
					result.Set(math.Exp(t.At(bi, i, j)-maxVal), bi, i, j)
					sum += result.At(bi, i, j)
				}
				// Normalize
				for j := 0; j < n; j++ {
					result.Set(result.At(bi, i, j)/sum, bi, i, j)
				}
			}
		}
	} else {
		panic("Softmax only supports 2D and 3D tensors")
	}

	return result
}

// ReLU applies ReLU activation element-wise.
func (t *Tensor) ReLU() *Tensor {
	result := t.Clone()
	for i := range result.Data {
		if result.Data[i] < 0 {
			result.Data[i] = 0
		}
	}
	return result
}

// GELU applies GELU activation (Gaussian Error Linear Unit).
func (t *Tensor) GELU() *Tensor {
	result := t.Clone()
	for i := range result.Data {
		x := result.Data[i]
		// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		result.Data[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
	}
	return result
}

// Sqrt applies element-wise square root.
func (t *Tensor) Sqrt() *Tensor {
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] = math.Sqrt(result.Data[i])
	}
	return result
}

// Mean computes the mean of all elements.
func (t *Tensor) Mean() float64 {
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return sum / float64(len(t.Data))
}

// Variance computes the variance of all elements.
func (t *Tensor) Variance() float64 {
	mean := t.Mean()
	sum := 0.0
	for _, v := range t.Data {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(t.Data))
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, data=%v)", t.Shape, t.Data[:min(10, len(t.Data))])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
