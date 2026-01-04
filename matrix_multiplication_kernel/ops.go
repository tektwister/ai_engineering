package matmul

import (
	"math"
	"math/rand"
)

// Add performs element-wise addition: C = A + B.
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrix dimensions must match for addition")
	}
	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	return result
}

// Sub performs element-wise subtraction: C = A - B.
func Sub(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrix dimensions must match for subtraction")
	}
	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] - b.Data[i]
	}
	return result
}

// Scale multiplies all elements by a scalar.
func Scale(m *Matrix, scalar float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		result.Data[i] = m.Data[i] * scalar
	}
	return result
}

// Hadamard performs element-wise multiplication (Hadamard product): C = A ⊙ B.
func Hadamard(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrix dimensions must match for Hadamard product")
	}
	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] * b.Data[i]
	}
	return result
}

// Dot computes the dot product of two vectors (1D matrices or flattened).
func Dot(a, b *Matrix) float64 {
	if len(a.Data) != len(b.Data) {
		panic("vectors must have the same length for dot product")
	}
	sum := 0.0
	for i := 0; i < len(a.Data); i++ {
		sum += a.Data[i] * b.Data[i]
	}
	return sum
}

// Sum returns the sum of all elements.
func Sum(m *Matrix) float64 {
	sum := 0.0
	for _, v := range m.Data {
		sum += v
	}
	return sum
}

// Mean returns the mean of all elements.
func Mean(m *Matrix) float64 {
	return Sum(m) / float64(len(m.Data))
}

// Max returns the maximum element.
func Max(m *Matrix) float64 {
	maxVal := math.Inf(-1)
	for _, v := range m.Data {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// Min returns the minimum element.
func Min(m *Matrix) float64 {
	minVal := math.Inf(1)
	for _, v := range m.Data {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

// ArgMax returns the index of the maximum element.
func ArgMax(m *Matrix) int {
	maxIdx := 0
	maxVal := m.Data[0]
	for i, v := range m.Data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// ArgMin returns the index of the minimum element.
func ArgMin(m *Matrix) int {
	minIdx := 0
	minVal := m.Data[0]
	for i, v := range m.Data {
		if v < minVal {
			minVal = v
			minIdx = i
		}
	}
	return minIdx
}

// Norm computes the Frobenius norm (L2 norm) of the matrix.
func Norm(m *Matrix) float64 {
	sum := 0.0
	for _, v := range m.Data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// Apply applies a function to each element of the matrix.
func Apply(m *Matrix, fn func(float64) float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i, v := range m.Data {
		result.Data[i] = fn(v)
	}
	return result
}

// Randn creates a matrix with random values from standard normal distribution.
func Randn(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = rand.NormFloat64()
	}
	return m
}

// RandnScaled creates a matrix with random values scaled by a factor.
func RandnScaled(scale float64, rows, cols int) *Matrix {
	m := Randn(rows, cols)
	for i := range m.Data {
		m.Data[i] *= scale
	}
	return m
}

// Rand creates a matrix with random values from uniform distribution [0, 1).
func Rand(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = rand.Float64()
	}
	return m
}

// RandRange creates a matrix with random values from uniform distribution [low, high).
func RandRange(low, high float64, rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	scale := high - low
	for i := range m.Data {
		m.Data[i] = rand.Float64()*scale + low
	}
	return m
}

// RowSum returns a column vector containing the sum of each row.
func RowSum(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, 1)
	for i := 0; i < m.Rows; i++ {
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			sum += m.At(i, j)
		}
		result.Data[i] = sum
	}
	return result
}

// ColSum returns a row vector containing the sum of each column.
func ColSum(m *Matrix) *Matrix {
	result := NewMatrix(1, m.Cols)
	for j := 0; j < m.Cols; j++ {
		sum := 0.0
		for i := 0; i < m.Rows; i++ {
			sum += m.At(i, j)
		}
		result.Data[j] = sum
	}
	return result
}

// RowMean returns a column vector containing the mean of each row.
func RowMean(m *Matrix) *Matrix {
	result := RowSum(m)
	for i := range result.Data {
		result.Data[i] /= float64(m.Cols)
	}
	return result
}

// ColMean returns a row vector containing the mean of each column.
func ColMean(m *Matrix) *Matrix {
	result := ColSum(m)
	for i := range result.Data {
		result.Data[i] /= float64(m.Rows)
	}
	return result
}

// Softmax applies the softmax function along rows.
func Softmax(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		// Find max for numerical stability
		maxVal := m.At(i, 0)
		for j := 1; j < m.Cols; j++ {
			if m.At(i, j) > maxVal {
				maxVal = m.At(i, j)
			}
		}

		// Compute exp and sum
		expSum := 0.0
		for j := 0; j < m.Cols; j++ {
			exp := math.Exp(m.At(i, j) - maxVal)
			result.Set(i, j, exp)
			expSum += exp
		}

		// Normalize
		for j := 0; j < m.Cols; j++ {
			result.Set(i, j, result.At(i, j)/expSum)
		}
	}
	return result
}

// ReLU applies the ReLU activation function element-wise.
func ReLU(m *Matrix) *Matrix {
	return Apply(m, func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

// Sigmoid applies the sigmoid activation function element-wise.
func Sigmoid(m *Matrix) *Matrix {
	return Apply(m, func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	})
}

// Tanh applies the tanh activation function element-wise.
func Tanh(m *Matrix) *Matrix {
	return Apply(m, math.Tanh)
}

// Exp applies the exponential function element-wise.
func Exp(m *Matrix) *Matrix {
	return Apply(m, math.Exp)
}

// Log applies the natural logarithm element-wise.
func Log(m *Matrix) *Matrix {
	return Apply(m, math.Log)
}

// Sqrt applies the square root element-wise.
func Sqrt(m *Matrix) *Matrix {
	return Apply(m, math.Sqrt)
}

// Pow raises each element to a power.
func Pow(m *Matrix, power float64) *Matrix {
	return Apply(m, func(x float64) float64 {
		return math.Pow(x, power)
	})
}

// Abs applies the absolute value element-wise.
func Abs(m *Matrix) *Matrix {
	return Apply(m, math.Abs)
}
