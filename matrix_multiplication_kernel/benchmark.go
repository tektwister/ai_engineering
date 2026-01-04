package matmul

import (
	"fmt"
	"runtime"
	"time"
)

// BenchmarkResult holds the result of a single benchmark run.
type BenchmarkResult struct {
	Kernel     Kernel
	Size       int
	Duration   time.Duration
	GFLOPS     float64 // Giga Floating Point Operations Per Second
	Iterations int
}

// String returns a formatted string representation of the result.
func (r BenchmarkResult) String() string {
	return fmt.Sprintf("%-16s | %5dx%-5d | %12v | %8.2f GFLOPS | %d iters",
		r.Kernel, r.Size, r.Size, r.Duration, r.GFLOPS, r.Iterations)
}

// BenchmarkConfig configures the benchmark run.
type BenchmarkConfig struct {
	Sizes      []int    // Matrix sizes to test
	Kernels    []Kernel // Kernels to benchmark
	MinTime    time.Duration // Minimum time to run each benchmark
	WarmupRuns int      // Number of warmup runs
}

// DefaultBenchmarkConfig returns a reasonable default configuration.
func DefaultBenchmarkConfig() *BenchmarkConfig {
	return &BenchmarkConfig{
		Sizes:      []int{64, 128, 256, 512, 1024},
		Kernels:    []Kernel{Naive, NaiveIKJ, Blocked, BlockedParallel, TransposedB},
		MinTime:    time.Second,
		WarmupRuns: 2,
	}
}

// Benchmark runs benchmarks according to the configuration.
func Benchmark(config *BenchmarkConfig) []BenchmarkResult {
	results := make([]BenchmarkResult, 0, len(config.Sizes)*len(config.Kernels))

	for _, size := range config.Sizes {
		// Create random matrices
		a := Randn(size, size)
		b := Randn(size, size)

		for _, kernel := range config.Kernels {
			result := benchmarkKernel(a, b, kernel, config)
			results = append(results, result)
		}
	}

	return results
}

// benchmarkKernel benchmarks a single kernel.
func benchmarkKernel(a, b *Matrix, kernel Kernel, config *BenchmarkConfig) BenchmarkResult {
	size := a.Rows

	// Warmup runs
	for i := 0; i < config.WarmupRuns; i++ {
		Multiply(a, b, kernel)
	}

	// Force GC before timing
	runtime.GC()

	// Run benchmark
	iterations := 0
	start := time.Now()
	for time.Since(start) < config.MinTime {
		Multiply(a, b, kernel)
		iterations++
	}
	elapsed := time.Since(start)

	// Calculate GFLOPS
	// Matrix multiplication of (n×n) × (n×n) requires 2n³ FLOPs
	// (n³ multiplications + n³ additions)
	flops := 2.0 * float64(size) * float64(size) * float64(size) * float64(iterations)
	gflops := flops / elapsed.Seconds() / 1e9

	return BenchmarkResult{
		Kernel:     kernel,
		Size:       size,
		Duration:   elapsed / time.Duration(iterations),
		GFLOPS:     gflops,
		Iterations: iterations,
	}
}

// BenchmarkSingle runs a benchmark for a specific size and kernel.
func BenchmarkSingle(size int, kernel Kernel) BenchmarkResult {
	a := Randn(size, size)
	b := Randn(size, size)
	config := &BenchmarkConfig{
		MinTime:    time.Second,
		WarmupRuns: 2,
	}
	return benchmarkKernel(a, b, kernel, config)
}

// PrintBenchmarkResults prints benchmark results in a formatted table.
func PrintBenchmarkResults(results []BenchmarkResult) {
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                    Matrix Multiplication Benchmark Results                │")
	fmt.Println("├──────────────────┬─────────────┬──────────────┬──────────────┬────────────┤")
	fmt.Println("│ Kernel           │ Size        │ Time/Op      │ GFLOPS       │ Iterations │")
	fmt.Println("├──────────────────┼─────────────┼──────────────┼──────────────┼────────────┤")

	for _, r := range results {
		fmt.Printf("│ %-16s │ %5dx%-5d │ %12v │ %10.2f   │ %10d │\n",
			r.Kernel, r.Size, r.Size, r.Duration, r.GFLOPS, r.Iterations)
	}

	fmt.Println("└──────────────────┴─────────────┴──────────────┴──────────────┴────────────┘")
}

// CompareKernels compares two kernels and returns the speedup factor.
func CompareKernels(size int, baseline, optimized Kernel) (baseResult, optResult BenchmarkResult, speedup float64) {
	a := Randn(size, size)
	b := Randn(size, size)
	config := &BenchmarkConfig{
		MinTime:    time.Second,
		WarmupRuns: 2,
	}

	baseResult = benchmarkKernel(a, b, baseline, config)
	optResult = benchmarkKernel(a, b, optimized, config)
	speedup = optResult.GFLOPS / baseResult.GFLOPS

	return
}

// VerifyKernelCorrectness checks if a kernel produces correct results.
func VerifyKernelCorrectness(kernel Kernel, tolerance float64) bool {
	// Test with small known matrices
	a := NewMatrixFromSlice([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	b := NewMatrixFromSlice([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})

	expected := NewMatrixFromSlice([][]float64{
		{58, 64},
		{139, 154},
	})

	result, err := Multiply(a, b, kernel)
	if err != nil {
		return false
	}

	return result.Equal(expected, tolerance)
}
