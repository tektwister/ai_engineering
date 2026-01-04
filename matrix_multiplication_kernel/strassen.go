package matmul

// StrassenThreshold is the size below which we fall back to standard multiplication.
// Strassen's algorithm has overhead that makes it slower for small matrices.
const StrassenThreshold = 128

// Strassen implements Strassen's algorithm for matrix multiplication.
// Time complexity: O(n^2.807) instead of O(n^3).
//
// The algorithm uses 7 multiplications instead of 8 for 2x2 blocks:
//
// For A = [A11 A12; A21 A22] and B = [B11 B12; B21 B22]:
//
//	M1 = (A11 + A22)(B11 + B22)
//	M2 = (A21 + A22)B11
//	M3 = A11(B12 - B22)
//	M4 = A22(B21 - B11)
//	M5 = (A11 + A12)B22
//	M6 = (A21 - A11)(B11 + B12)
//	M7 = (A12 - A22)(B21 + B22)
//
//	C11 = M1 + M4 - M5 + M7
//	C12 = M3 + M5
//	C21 = M2 + M4
//	C22 = M1 - M2 + M3 + M6
func Strassen(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, nil // Return nil for dimension mismatch
	}

	// Ensure matrices are square and power of 2 for simplicity
	n := maxDim(a.Rows, a.Cols, b.Rows, b.Cols)
	n = nextPowerOf2(n)

	// Pad matrices to n×n
	aPadded := padMatrix(a, n)
	bPadded := padMatrix(b, n)

	// Perform Strassen multiplication
	cPadded := strassenRecursive(aPadded, bPadded)

	// Extract the result (unpad)
	return extractSubmatrix(cPadded, a.Rows, b.Cols), nil
}

// strassenRecursive performs the recursive Strassen multiplication.
func strassenRecursive(a, b *Matrix) *Matrix {
	n := a.Rows

	// Base case: use standard multiplication for small matrices
	if n <= StrassenThreshold {
		return multiplyBlocked(a, b, DefaultBlockSize)
	}

	// Split matrices into quadrants
	half := n / 2

	a11 := getSubmatrix(a, 0, 0, half)
	a12 := getSubmatrix(a, 0, half, half)
	a21 := getSubmatrix(a, half, 0, half)
	a22 := getSubmatrix(a, half, half, half)

	b11 := getSubmatrix(b, 0, 0, half)
	b12 := getSubmatrix(b, 0, half, half)
	b21 := getSubmatrix(b, half, 0, half)
	b22 := getSubmatrix(b, half, half, half)

	// Calculate the 7 Strassen products
	m1 := strassenRecursive(matAdd(a11, a22), matAdd(b11, b22))     // M1 = (A11+A22)(B11+B22)
	m2 := strassenRecursive(matAdd(a21, a22), b11)                  // M2 = (A21+A22)B11
	m3 := strassenRecursive(a11, matSub(b12, b22))                  // M3 = A11(B12-B22)
	m4 := strassenRecursive(a22, matSub(b21, b11))                  // M4 = A22(B21-B11)
	m5 := strassenRecursive(matAdd(a11, a12), b22)                  // M5 = (A11+A12)B22
	m6 := strassenRecursive(matSub(a21, a11), matAdd(b11, b12))     // M6 = (A21-A11)(B11+B12)
	m7 := strassenRecursive(matSub(a12, a22), matAdd(b21, b22))     // M7 = (A12-A22)(B21+B22)

	// Calculate result quadrants
	c11 := matAdd(matSub(matAdd(m1, m4), m5), m7) // C11 = M1+M4-M5+M7
	c12 := matAdd(m3, m5)                          // C12 = M3+M5
	c21 := matAdd(m2, m4)                          // C21 = M2+M4
	c22 := matAdd(matSub(matAdd(m1, m3), m2), m6) // C22 = M1-M2+M3+M6

	// Combine quadrants into result
	return combineQuadrants(c11, c12, c21, c22)
}

// matAdd adds two matrices element-wise.
func matAdd(a, b *Matrix) *Matrix {
	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	return result
}

// matSub subtracts two matrices element-wise.
func matSub(a, b *Matrix) *Matrix {
	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] - b.Data[i]
	}
	return result
}

// getSubmatrix extracts a square submatrix of given size starting at (rowStart, colStart).
func getSubmatrix(m *Matrix, rowStart, colStart, size int) *Matrix {
	result := NewMatrix(size, size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			result.Set(i, j, m.At(rowStart+i, colStart+j))
		}
	}
	return result
}

// padMatrix pads a matrix to n×n with zeros.
func padMatrix(m *Matrix, n int) *Matrix {
	if m.Rows == n && m.Cols == n {
		return m.Clone()
	}
	result := NewMatrix(n, n)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Set(i, j, m.At(i, j))
		}
	}
	return result
}

// extractSubmatrix extracts a submatrix of specified dimensions from the top-left corner.
func extractSubmatrix(m *Matrix, rows, cols int) *Matrix {
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, m.At(i, j))
		}
	}
	return result
}

// combineQuadrants combines four quadrants into a single matrix.
func combineQuadrants(c11, c12, c21, c22 *Matrix) *Matrix {
	half := c11.Rows
	n := half * 2
	result := NewMatrix(n, n)

	for i := 0; i < half; i++ {
		for j := 0; j < half; j++ {
			result.Set(i, j, c11.At(i, j))
			result.Set(i, j+half, c12.At(i, j))
			result.Set(i+half, j, c21.At(i, j))
			result.Set(i+half, j+half, c22.At(i, j))
		}
	}
	return result
}

// nextPowerOf2 returns the smallest power of 2 >= n.
func nextPowerOf2(n int) int {
	if n <= 1 {
		return 1
	}
	power := 1
	for power < n {
		power *= 2
	}
	return power
}

// maxDim returns the maximum of multiple dimensions.
func maxDim(dims ...int) int {
	maxVal := dims[0]
	for _, d := range dims[1:] {
		if d > maxVal {
			maxVal = d
		}
	}
	return maxVal
}
