// Package matmul provides optimized matrix multiplication kernels.
// It implements various algorithms from naive O(n³) to cache-optimized blocked versions.
package matmul

import (
	"fmt"
)

// Matrix represents a 2D matrix stored in row-major order.
// The underlying data is stored as a flat slice for cache efficiency.
type Matrix struct {
	Data []float64
	Rows int
	Cols int
}

// NewMatrix creates a new matrix with the specified dimensions.
// All elements are initialized to zero.
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Data: make([]float64, rows*cols),
		Rows: rows,
		Cols: cols,
	}
}

// NewMatrixFromSlice creates a matrix from a 2D slice.
func NewMatrixFromSlice(data [][]float64) *Matrix {
	if len(data) == 0 {
		return NewMatrix(0, 0)
	}
	rows := len(data)
	cols := len(data[0])
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Set(i, j, data[i][j])
		}
	}
	return m
}

// NewMatrixFromFlat creates a matrix from a flat slice.
func NewMatrixFromFlat(data []float64, rows, cols int) *Matrix {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("data length %d doesn't match dimensions %dx%d", len(data), rows, cols))
	}
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)
	return &Matrix{
		Data: dataCopy,
		Rows: rows,
		Cols: cols,
	}
}

// Index returns the flat index for the given row and column.
func (m *Matrix) Index(row, col int) int {
	return row*m.Cols + col
}

// At returns the element at position (row, col).
func (m *Matrix) At(row, col int) float64 {
	return m.Data[m.Index(row, col)]
}

// Set sets the element at position (row, col).
func (m *Matrix) Set(row, col int, value float64) {
	m.Data[m.Index(row, col)] = value
}

// Clone creates a deep copy of the matrix.
func (m *Matrix) Clone() *Matrix {
	dataCopy := make([]float64, len(m.Data))
	copy(dataCopy, m.Data)
	return &Matrix{
		Data: dataCopy,
		Rows: m.Rows,
		Cols: m.Cols,
	}
}

// Shape returns the dimensions of the matrix.
func (m *Matrix) Shape() (int, int) {
	return m.Rows, m.Cols
}

// To2D converts the matrix to a 2D slice representation.
func (m *Matrix) To2D() [][]float64 {
	result := make([][]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		result[i] = make([]float64, m.Cols)
		for j := 0; j < m.Cols; j++ {
			result[i][j] = m.At(i, j)
		}
	}
	return result
}

// Transpose returns the transpose of the matrix.
func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Set(j, i, m.At(i, j))
		}
	}
	return result
}

// String returns a string representation of the matrix.
func (m *Matrix) String() string {
	result := fmt.Sprintf("Matrix(%dx%d):\n", m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		result += "["
		for j := 0; j < m.Cols; j++ {
			if j > 0 {
				result += ", "
			}
			result += fmt.Sprintf("%.4f", m.At(i, j))
		}
		result += "]\n"
	}
	return result
}

// Equal checks if two matrices are equal within a tolerance.
func (m *Matrix) Equal(other *Matrix, tolerance float64) bool {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return false
	}
	for i := 0; i < len(m.Data); i++ {
		diff := m.Data[i] - other.Data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			return false
		}
	}
	return true
}

// Zeros creates a matrix filled with zeros.
func Zeros(rows, cols int) *Matrix {
	return NewMatrix(rows, cols)
}

// Ones creates a matrix filled with ones.
func Ones(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = 1.0
	}
	return m
}

// Eye creates an identity matrix.
func Eye(n int) *Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m.Set(i, i, 1.0)
	}
	return m
}
