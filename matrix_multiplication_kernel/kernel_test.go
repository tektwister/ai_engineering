package matmul

import (
	"fmt"
	"math"
	"testing"
)

// TestMatrixCreation tests matrix creation and basic operations.
func TestMatrixCreation(t *testing.T) {
	t.Run("NewMatrix", func(t *testing.T) {
		m := NewMatrix(3, 4)
		if m.Rows != 3 || m.Cols != 4 {
			t.Errorf("Expected 3x4 matrix, got %dx%d", m.Rows, m.Cols)
		}
		if len(m.Data) != 12 {
			t.Errorf("Expected 12 elements, got %d", len(m.Data))
		}
	})

	t.Run("NewMatrixFromSlice", func(t *testing.T) {
		data := [][]float64{
			{1, 2, 3},
			{4, 5, 6},
		}
		m := NewMatrixFromSlice(data)
		if m.Rows != 2 || m.Cols != 3 {
			t.Errorf("Expected 2x3 matrix, got %dx%d", m.Rows, m.Cols)
		}
		if m.At(0, 2) != 3 || m.At(1, 1) != 5 {
			t.Error("Matrix values not set correctly")
		}
	})

	t.Run("Eye", func(t *testing.T) {
		m := Eye(3)
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				expected := 0.0
				if i == j {
					expected = 1.0
				}
				if m.At(i, j) != expected {
					t.Errorf("Eye(%d,%d): expected %f, got %f", i, j, expected, m.At(i, j))
				}
			}
		}
	})

	t.Run("Transpose", func(t *testing.T) {
		m := NewMatrixFromSlice([][]float64{
			{1, 2, 3},
			{4, 5, 6},
		})
		mt := m.Transpose()
		if mt.Rows != 3 || mt.Cols != 2 {
			t.Errorf("Expected 3x2 transpose, got %dx%d", mt.Rows, mt.Cols)
		}
		if mt.At(0, 0) != 1 || mt.At(0, 1) != 4 || mt.At(2, 1) != 6 {
			t.Error("Transpose values incorrect")
		}
	})
}

// TestNaiveMultiplication tests the naive O(n³) multiplication.
func TestNaiveMultiplication(t *testing.T) {
	a := NewMatrixFromSlice([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	b := NewMatrixFromSlice([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})

	expected := [][]float64{
		{58, 64},
		{139, 154},
	}

	result, err := Multiply(a, b, Naive)
	if err != nil {
		t.Fatalf("Multiply failed: %v", err)
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if result.At(i, j) != expected[i][j] {
				t.Errorf("Result[%d][%d]: expected %f, got %f", i, j, expected[i][j], result.At(i, j))
			}
		}
	}
}

// TestAllKernels verifies that all kernels produce the same result.
func TestAllKernels(t *testing.T) {
	kernels := []Kernel{Naive, NaiveIKJ, Blocked, BlockedParallel, TransposedB}

	for _, size := range []int{16, 33, 64, 100} {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			a := RandnScaled(1.0, size, size)
			b := RandnScaled(1.0, size, size)

			// Get reference result from Naive
			reference, _ := Multiply(a, b, Naive)

			for _, kernel := range kernels[1:] {
				t.Run(kernel.String(), func(t *testing.T) {
					result, err := Multiply(a, b, kernel)
					if err != nil {
						t.Fatalf("Kernel %s failed: %v", kernel, err)
					}
					if !result.Equal(reference, 1e-9) {
						t.Errorf("Kernel %s produced different result", kernel)
					}
				})
			}
		})
	}
}

// TestNonSquareMatrices tests multiplication of non-square matrices.
func TestNonSquareMatrices(t *testing.T) {
	testCases := []struct {
		aRows, aCols int
		bRows, bCols int
	}{
		{2, 3, 3, 4},
		{1, 5, 5, 1},
		{10, 20, 20, 15},
		{7, 11, 11, 13},
	}

	kernels := []Kernel{Naive, NaiveIKJ, Blocked, BlockedParallel, TransposedB}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%dx%d_x_%dx%d", tc.aRows, tc.aCols, tc.bRows, tc.bCols), func(t *testing.T) {
			a := RandnScaled(1.0, tc.aRows, tc.aCols)
			b := RandnScaled(1.0, tc.bRows, tc.bCols)

			reference, _ := Multiply(a, b, Naive)

			for _, kernel := range kernels[1:] {
				result, err := Multiply(a, b, kernel)
				if err != nil {
					t.Fatalf("Kernel %s failed: %v", kernel, err)
				}
				if result.Rows != tc.aRows || result.Cols != tc.bCols {
					t.Errorf("Kernel %s: wrong dimensions %dx%d", kernel, result.Rows, result.Cols)
				}
				if !result.Equal(reference, 1e-9) {
					t.Errorf("Kernel %s produced different result", kernel)
				}
			}
		})
	}
}

// TestDimensionMismatch tests that incompatible dimensions produce an error.
func TestDimensionMismatch(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(4, 5) // 3 != 4, should fail

	_, err := Multiply(a, b, Naive)
	if err == nil {
		t.Error("Expected error for dimension mismatch, got nil")
	}
}

// TestIdentityMultiplication tests that A × I = A.
func TestIdentityMultiplication(t *testing.T) {
	for _, size := range []int{10, 50, 100} {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			a := RandnScaled(1.0, size, size)
			i := Eye(size)

			result, _ := Multiply(a, i, BlockedParallel)
			if !result.Equal(a, 1e-9) {
				t.Error("A × I != A")
			}
		})
	}
}

// TestStrassen tests Strassen's algorithm.
func TestStrassen(t *testing.T) {
	for _, size := range []int{16, 32, 64, 128, 200, 256} {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			a := RandnScaled(0.1, size, size)
			b := RandnScaled(0.1, size, size)

			reference, _ := Multiply(a, b, Naive)
			result, err := Strassen(a, b)
			if err != nil {
				t.Fatalf("Strassen failed: %v", err)
			}

			// Strassen has slightly lower precision due to more operations
			if !result.Equal(reference, 1e-6) {
				t.Error("Strassen produced different result than Naive")
			}
		})
	}
}

// TestOps tests additional matrix operations.
func TestOps(t *testing.T) {
	t.Run("Add", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{1, 2}, {3, 4}})
		b := NewMatrixFromSlice([][]float64{{5, 6}, {7, 8}})
		c := Add(a, b)
		expected := NewMatrixFromSlice([][]float64{{6, 8}, {10, 12}})
		if !c.Equal(expected, 1e-9) {
			t.Error("Add failed")
		}
	})

	t.Run("Sub", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{5, 6}, {7, 8}})
		b := NewMatrixFromSlice([][]float64{{1, 2}, {3, 4}})
		c := Sub(a, b)
		expected := NewMatrixFromSlice([][]float64{{4, 4}, {4, 4}})
		if !c.Equal(expected, 1e-9) {
			t.Error("Sub failed")
		}
	})

	t.Run("Scale", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{1, 2}, {3, 4}})
		c := Scale(a, 2.0)
		expected := NewMatrixFromSlice([][]float64{{2, 4}, {6, 8}})
		if !c.Equal(expected, 1e-9) {
			t.Error("Scale failed")
		}
	})

	t.Run("Hadamard", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{1, 2}, {3, 4}})
		b := NewMatrixFromSlice([][]float64{{5, 6}, {7, 8}})
		c := Hadamard(a, b)
		expected := NewMatrixFromSlice([][]float64{{5, 12}, {21, 32}})
		if !c.Equal(expected, 1e-9) {
			t.Error("Hadamard failed")
		}
	})

	t.Run("Norm", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{3, 4}})
		norm := Norm(a)
		if math.Abs(norm-5.0) > 1e-9 {
			t.Errorf("Norm expected 5, got %f", norm)
		}
	})

	t.Run("Softmax", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{1, 2, 3}})
		s := Softmax(a)
		sum := 0.0
		for j := 0; j < 3; j++ {
			sum += s.At(0, j)
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("Softmax sum expected 1, got %f", sum)
		}
		// Values should be in increasing order
		if s.At(0, 0) >= s.At(0, 1) || s.At(0, 1) >= s.At(0, 2) {
			t.Error("Softmax values not in expected order")
		}
	})

	t.Run("ReLU", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{-1, 0, 1, 2}})
		r := ReLU(a)
		expected := NewMatrixFromSlice([][]float64{{0, 0, 1, 2}})
		if !r.Equal(expected, 1e-9) {
			t.Error("ReLU failed")
		}
	})

	t.Run("Sigmoid", func(t *testing.T) {
		a := NewMatrixFromSlice([][]float64{{0}})
		s := Sigmoid(a)
		if math.Abs(s.At(0, 0)-0.5) > 1e-9 {
			t.Errorf("Sigmoid(0) expected 0.5, got %f", s.At(0, 0))
		}
	})
}

// TestRowColOperations tests row and column operations.
func TestRowColOperations(t *testing.T) {
	m := NewMatrixFromSlice([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	t.Run("RowSum", func(t *testing.T) {
		rs := RowSum(m)
		if rs.At(0, 0) != 6 || rs.At(1, 0) != 15 {
			t.Errorf("RowSum incorrect: got [%f, %f]", rs.At(0, 0), rs.At(1, 0))
		}
	})

	t.Run("ColSum", func(t *testing.T) {
		cs := ColSum(m)
		if cs.At(0, 0) != 5 || cs.At(0, 1) != 7 || cs.At(0, 2) != 9 {
			t.Errorf("ColSum incorrect: got [%f, %f, %f]", cs.At(0, 0), cs.At(0, 1), cs.At(0, 2))
		}
	})

	t.Run("RowMean", func(t *testing.T) {
		rm := RowMean(m)
		if rm.At(0, 0) != 2 || rm.At(1, 0) != 5 {
			t.Errorf("RowMean incorrect: got [%f, %f]", rm.At(0, 0), rm.At(1, 0))
		}
	})
}
