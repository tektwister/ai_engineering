package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	matmul "github.com/tektwister/ai_engineering/matrix_multiplication_kernel"
)

func main() {
	// Parse command-line flags
	size := flag.Int("size", 512, "Matrix size (NxN)")
	verify := flag.Bool("verify", false, "Verify correctness of all kernels")
	quick := flag.Bool("quick", false, "Run quick benchmark with fewer iterations")
	flag.Parse()

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           Matrix Multiplication Kernel Benchmark                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ CPUs: %-6d | Matrix Size: %4dx%-4d | Go Version: %-20s  ║\n",
		runtime.NumCPU(), *size, *size, runtime.Version())
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	if *verify {
		verifyAllKernels()
		return
	}

	runBenchmark(*size, *quick)
}

func verifyAllKernels() {
	fmt.Println("Verifying kernel correctness...")
	fmt.Println()

	kernels := []matmul.Kernel{
		matmul.Naive,
		matmul.NaiveIKJ,
		matmul.Blocked,
		matmul.BlockedParallel,
		matmul.TransposedB,
	}

	allPassed := true
	for _, kernel := range kernels {
		if matmul.VerifyKernelCorrectness(kernel, 1e-9) {
			fmt.Printf("  ✓ %-16s PASSED\n", kernel)
		} else {
			fmt.Printf("  ✗ %-16s FAILED\n", kernel)
			allPassed = false
		}
	}

	fmt.Println()
	if allPassed {
		fmt.Println("All kernels verified successfully!")
	} else {
		fmt.Println("Some kernels failed verification!")
		os.Exit(1)
	}
}

func runBenchmark(size int, quick bool) {
	fmt.Println("Running benchmarks...")
	fmt.Println()

	config := matmul.DefaultBenchmarkConfig()
	config.Sizes = []int{size}
	if quick {
		config.MinTime = config.MinTime / 4
		config.WarmupRuns = 1
	}

	results := matmul.Benchmark(config)
	matmul.PrintBenchmarkResults(results)

	// Find best kernel
	var best matmul.BenchmarkResult
	for _, r := range results {
		if r.GFLOPS > best.GFLOPS {
			best = r
		}
	}

	fmt.Println()
	fmt.Printf("🏆 Best Kernel: %s (%.2f GFLOPS)\n", best.Kernel, best.GFLOPS)

	// Show speedup vs naive
	var naive matmul.BenchmarkResult
	for _, r := range results {
		if r.Kernel == matmul.Naive {
			naive = r
			break
		}
	}

	if naive.GFLOPS > 0 {
		speedup := best.GFLOPS / naive.GFLOPS
		fmt.Printf("📈 Speedup vs Naive: %.2fx\n", speedup)
	}

	// Also test Strassen for comparison if size is large enough
	if size >= 256 {
		fmt.Println()
		fmt.Println("Testing Strassen's algorithm...")

		a := matmul.RandnScaled(1.0, size, size)
		b := matmul.RandnScaled(1.0, size, size)

		strassenResult, _ := matmul.Strassen(a, b)
		reference, _ := matmul.Multiply(a, b, matmul.Naive)

		if strassenResult.Equal(reference, 1e-6) {
			fmt.Println("  ✓ Strassen result matches reference")
		} else {
			fmt.Println("  ✗ Strassen result differs (may be precision issues)")
		}
	}
}
