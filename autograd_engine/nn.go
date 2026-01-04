package autograd

import (
	"math/rand"
)

// Module is the interface for all neural network modules.
type Module interface {
	Parameters() []*Value
	ZeroGrad()
}

// Neuron represents a single neuron with weights and a bias.
type Neuron struct {
	w   []*Value
	b   *Value
	act bool // non-linearity
}

// NewNeuron creates a new Neuron with nin inputs.
// If nonLin is true, applies ReLU activation.
func NewNeuron(nin int, nonLin bool) *Neuron {
	w := make([]*Value, nin)
	for i := range w {
		w[i] = NewValue(rand.Float64()*2 - 1) // random weights between -1 and 1
	}
	b := NewValue(0) // bias initialized to 0
	return &Neuron{w: w, b: b, act: nonLin}
}

// Call computes the output of the neuron for input x.
func (n *Neuron) Call(x []*Value) *Value {
	if len(x) != len(n.w) {
		panic("input size mismatch")
	}
	
	act := n.b
	for i, wi := range n.w {
		act = act.Add(wi.Mul(x[i]))
	}
	
	if n.act {
		return act.ReLU()
	}
	return act
}

// Parameters returns the parameters (weights + bias) of the neuron.
func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, len(n.w)+1)
	copy(params, n.w)
	params[len(n.w)] = n.b
	return params
}

// ZeroGrad resets gradients of all parameters in the neuron.
func (n *Neuron) ZeroGrad() {
	for _, p := range n.Parameters() {
		p.ZeroGrad()
	}
}

// Layer represents a layer of neurons.
type Layer struct {
	neurons []*Neuron
}

// NewLayer creates a new Layer with nin inputs and nout outputs.
func NewLayer(nin, nout int, nonLin bool) *Layer {
	neurons := make([]*Neuron, nout)
	for i := range neurons {
		neurons[i] = NewNeuron(nin, nonLin)
	}
	return &Layer{neurons: neurons}
}

// Call computes the output of the layer for input x.
// Returns a slice of Values.
func (l *Layer) Call(x []*Value) []*Value {
	outs := make([]*Value, len(l.neurons))
	for i, n := range l.neurons {
		outs[i] = n.Call(x)
	}
	return outs
}

// Parameters returns the parameters of all neurons in the layer.
func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

// ZeroGrad resets gradients of all parameters in the layer.
func (l *Layer) ZeroGrad() {
	for _, p := range l.Parameters() {
		p.ZeroGrad()
	}
}

// MLP represents a Multi-Layer Perceptron.
type MLP struct {
	layers []*Layer
}

// NewMLP creates a new MLP.
// nin is the number of inputs.
// nouts is a list of the number of neurons in each layer.
func NewMLP(nin int, nouts []int) *MLP {
	layers := make([]*Layer, len(nouts))
	sz := append([]int{nin}, nouts...)
	
	for i := 0; i < len(nouts); i++ {
		// Apply ReLU to all layers except possibly the last one if you want linear output, 
		// but micrograd usually puts non-linearity on all layers in constructor inputs or assumes it.
		// Micrograd implementation: 
		// sz = [nin] + nouts
		// layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
		// So last layer is linear.
		nonLin := i != len(nouts)-1
		layers[i] = NewLayer(sz[i], sz[i+1], nonLin)
	}
	return &MLP{layers: layers}
}

// Call computes the output of the MLP for input x.
func (m *MLP) Call(x []*Value) []*Value {
	for _, l := range m.layers {
		x = l.Call(x)
	}
	return x
}

// Parameters returns the parameters of all layers in the MLP.
func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, l := range m.layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

// ZeroGrad resets gradients of all parameters in the MLP.
func (m *MLP) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}
