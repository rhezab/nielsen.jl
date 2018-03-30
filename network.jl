mutable struct Network
    num_layers::Int
    sizes::Array
    biases::Array # of vectors
    weights::Array # of matrices
end

# methods on Network struct
function init(sizes::Vector)
    num_layers = length(sizes)
    biases = [randn(ns) for ns in sizes[2:end]] # (ns,1) ? 
    weights = [randn((sizes[ns],sizes[ns-1])) for ns in 2:num_layers]
    Network(num_layers, sizes, biases, weights)
end

function feedforward(net::Network, a::Vector; fpass=false)
    if fpass
        as = [a]
        zs = []
    end
    for (b,w) in zip(net.biases, net.weights)
        z = w*a + b
        a = sigmoid(z)
        if fpass
            push!(as, a)
            push!(zs, z)
        end
    end
    if fpass
        return a, as, zs
    else
        return a
    end
end

function SGD!(net::Network, training_data, epochs, mini_batch_size, 
    learning_rate; test_data = [])
    n_test = length(test_data)
    n = length(training_data)
    for i in 1:epochs
        shuffle!(training_data)
        mini_batches = [training_data[k:k+mini_batch_size-1] for 
            k in 1:mini_batch_size:(n-mini_batch_size)]
        for mini_batch in mini_batches
            update_mini_batch!(net, mini_batch, learning_rate)
        end
        if n_test > 0
            acc = evaluate(net, test_data)
            println("Epoch $i: $acc")
        else
            println("Epoch $i")
        end
    end
end

function update_mini_batch!(net::Network, mini_batch, learning_rate)
    grad_b = [zeros(size(b)) for b in net.biases]
    grad_w = [zeros(size(w)) for w in net.weights]
    for (x,y) in mini_batch
        (delta_grad_b, delta_grad_w) = backprop(net, x,y)
        grad_b += delta_grad_b
        grad_w += delta_grad_w
    end
    mini_batch_size = length(mini_batch)
    net.biases -= (learning_rate/mini_batch_size) * grad_b 
    net.weights -= (learning_rate/mini_batch_size) * grad_w 
end

function backprop(net::Network, x::Vector, y::Vector)
    grad_b = [zeros(size(b)) for b in net.biases]
    grad_w = [zeros(size(w)) for w in net.weights]
    # feedforward
    (output, as, zs) = feedforward(net, x, fpass=true)
    # backward pass
    error = grad_cost(output, y) .* sigmoid_prime(zs[end])
    grad_b[end] = error
    grad_w[end] = error * as[end-1].' # '
    for l in 1:(net.num_layers-2)
        z = zs[end-l]
        sp = sigmoid_prime(z)
        error = (net.weights[end-l+1].' * error) .* sp # '
        grad_b[end-l] = error
        grad_w[end-l] = error * as[end-l-1].' # '
    end
    (grad_b, grad_w)
end

function evaluate(net::Network, test_data::Array)
    test_results = [(findmax(feedforward(net, x))[2]-1,y) for (x,y) in test_data]
    correct = sum(Int(x==y) for (x,y) in test_results)
    n = length(test_results)
    acc = correct / n
end

function vectorize_result(label)
    y = zeros(10)
    y[Int(label) + 1] = 1.0
    y
end

# Helpers
function grad_cost(output_activation::Array, y::Array)
    output_activation - y
end

function sigmoid(z::Number)
    1 / (1+e^(-z))
end

function sigmoid(z::Array)
    broadcast(sigmoid, z)
end

function sigmoid_prime(z::Number)
    sigmoid(z)*(1-sigmoid(z))
end

function sigmoid_prime(z::Array)
    broadcast(sigmoid_prime, z)
end

# Testing Helpers
using MNIST

function load_data()
    train_data = [(trainfeatures(i) / 255.0, vectorize_result(trainlabel(i))) for 
        i in 1:50000]
    validation_data = [(trainfeatures(i) / 255.0, trainlabel(i)) for 
        i in 50001:60000]
    test_data = [(testfeatures(i) / 255.0, testlabel(i)) for 
        i in 1:10000]
    (train_data, validation_data, test_data)
end

function test_mnist()
    train_data, validation_data, test_data = load_data()
    net = init([784,30,10])
    SGD!(net, train_data, 30, 10, 3.0, test_data=test_data)
    net
end
