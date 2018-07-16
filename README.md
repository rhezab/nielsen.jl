Julia implementations of the networks described in Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). `Pkg.add("MNIST")` in the Julia interpreter to get access to the MNIST dataset.

Here are my notes for the first network implemented in `network.jl`: [The Hello World Network](https://paper.dropbox.com/doc/The-Hello-World-Network--ACwvSXz585Fa9lDqft~ZD2M3AQ-kjsxv5Kj9PxhdDZbI1rAh), if you will.

`network.jl` is a really barebones neural network with:
- Initialize weights and biases using a Gaussian with mean=0 and stdv=1
- Sigmoid activation function for the neurons
- Quadratic cost function (no regularization)
- Backprop and SGD to update weights and biases

`network2.jl` implements some improvements (everything else is the same):
- Initialize weights using a Gaussian with mean=0 and stdv=1/sqrt(n_in) where n_in is number of weights connecting in to the neuron
- Cross-entropy cost function (with L2 regularization)
- Fully matrix based approach (as described in Chapter 2, be gone for loops!)
