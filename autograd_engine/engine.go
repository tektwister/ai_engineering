package autograd

import (
	"fmt"
	"math"
)

// Value represents a scalar value with support for automatic differentiation.
type Value struct {
	data     float64
	grad     float64
	prev     []*Value
	op       string
	backward func()
}

// NewValue creates a new Value scalar.
func NewValue(data float64) *Value {
	return &Value{
		data: data,
		grad: 0.0,
	}
}

// Data returns the underlying data of the Value.
func (v *Value) Data() float64 {
	return v.data
}

// SetData sets the underlying data of the Value.
func (v *Value) SetData(d float64) {
	v.data = d
}

// Grad returns the gradient of the Value.
func (v *Value) Grad() float64 {
	return v.grad
}

// ZeroGrad resets the gradient to 0.
func (v *Value) ZeroGrad() {
	v.grad = 0.0
}

// Add performs addition: v + other
func (v *Value) Add(other *Value) *Value {
	out := &Value{
		data: v.data + other.data,
		prev: []*Value{v, other},
		op:   "+",
	}

	out.backward = func() {
		v.grad += 1.0 * out.grad
		other.grad += 1.0 * out.grad
	}
	return out
}

// AddScalar performs addition with a float64: v + scalar
func (v *Value) AddScalar(scalar float64) *Value {
	other := NewValue(scalar)
	return v.Add(other)
}

// Mul performs multiplication: v * other
func (v *Value) Mul(other *Value) *Value {
	out := &Value{
		data: v.data * other.data,
		prev: []*Value{v, other},
		op:   "*",
	}

	out.backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}
	return out
}

// MulScalar performs multiplication with a float64: v * scalar
func (v *Value) MulScalar(scalar float64) *Value {
	other := NewValue(scalar)
	return v.Mul(other)
}

// Pow performs power operation: v ^ other
func (v *Value) Pow(other float64) *Value {
	out := &Value{
		data: math.Pow(v.data, other),
		prev: []*Value{v},
		op:   fmt.Sprintf("**%f", other),
	}

	out.backward = func() {
		v.grad += other * math.Pow(v.data, other-1) * out.grad
	}
	return out
}

// Neg computes -v
func (v *Value) Neg() *Value {
	return v.MulScalar(-1)
}

// Sub computes v - other
func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

// Tanh performs hyperbolic tangent activation
func (v *Value) Tanh() *Value {
	x := v.data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := &Value{
		data: t,
		prev: []*Value{v},
		op:   "tanh",
	}

	out.backward = func() {
		v.grad += (1 - t*t) * out.grad
	}
	return out
}

// ReLU performs Rectified Linear Unit activation
func (v *Value) ReLU() *Value {
	out := &Value{
		data: 0,
		prev: []*Value{v},
		op:   "relu",
	}
	if v.data > 0 {
		out.data = v.data
	}

	out.backward = func() {
		if v.data > 0 {
			v.grad += 1.0 * out.grad
		}
	}
	return out
}

// Backward computes the gradients for the graph using topological sort.
func (v *Value) Backward() {
	topo := []*Value{}
	visited := make(map[*Value]bool)

	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.prev {
				buildTopo(child)
			}
			topo = append(topo, node)
		}
	}
	buildTopo(v)

	v.grad = 1.0
	// Go in reverse order of topological sort
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		if node.backward != nil {
			node.backward()
		}
	}
}

// String implements the Stringer interface for pretty printing.
func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%f, grad=%f, op=%s)", v.data, v.grad, v.op)
}
