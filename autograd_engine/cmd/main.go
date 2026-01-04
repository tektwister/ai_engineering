package main

import (
	"fmt"
	"math/rand"
	"time"

	autograd "github.com/tektwister/ai_engineering/autograd_engine"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1. Create a tiny dataset
	// inputs
	xs := [][]*autograd.Value{
		{autograd.NewValue(2.0), autograd.NewValue(3.0), autograd.NewValue(-1.0)},
		{autograd.NewValue(3.0), autograd.NewValue(-1.0), autograd.NewValue(0.5)},
		{autograd.NewValue(0.5), autograd.NewValue(1.0), autograd.NewValue(1.0)},
		{autograd.NewValue(1.0), autograd.NewValue(1.0), autograd.NewValue(-1.0)},
	}
	// targets
	ys := []*autograd.Value{
		autograd.NewValue(1.0),
		autograd.NewValue(-1.0),
		autograd.NewValue(-1.0),
		autograd.NewValue(1.0),
	}

	// 2. Initialize the model
	// MLP with 3 inputs, two hidden layers of 4 neurons, and 1 output
	n := autograd.NewMLP(3, []int{4, 4, 1})

	learningRate := 0.05 // increased from typical 0.01 because non-batched or small
	iterations := 20

	fmt.Println("Starting training...")

	// 3. Training loop
	for k := 0; k < iterations; k++ {
		// Forward pass
		var loss *autograd.Value
		totalLoss := autograd.NewValue(0.0)

		for i, x := range xs {
			prediction := n.Call(x)[0]            // Get the single output
			diff := prediction.Sub(ys[i])        // (pred - target)
			sqErr := diff.Pow(2)                 // (pred - target)^2
			totalLoss = totalLoss.Add(sqErr)
		}
		
		// Mean Squared Error (optional normalisation, but sum is fine for small batch)
		loss = totalLoss

		// Zero gradients
		n.ZeroGrad()

		// Backward pass
		loss.Backward()

		// Update parameters (Gradient Descent)
		for _, p := range n.Parameters() {
			p.SetData(p.Data() - learningRate*p.Grad())
		}

		fmt.Printf("Step %d: loss %f\n", k, loss.Data())
	}
	
	// Check final predictions
	fmt.Println("\nFinal predictions:")
	for i, x := range xs {
		pred := n.Call(x)[0]
		fmt.Printf("Input: %v, Target: %f, Prediction: %f\n", i, ys[i].Data(), pred.Data())
	}
}
