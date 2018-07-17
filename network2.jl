# Cross Entropy Cost struct and methods
struct CrossEntropyCost
    fn
    delta
end

function cec_fn(a, y)
    -1 * sum((y * log(a) + (1-y) * ln(1-a)))
end

function cec_delta(z, a, y)
    a - y
end

function return_cec()
    CrossEntropyCost(cec_fn, cec_delta)
end

# Quadratic Cost struct and methods
struct QuadraticCost
    fn
    delta
end

function qc_fn(a, y)
    0.5 * norm(y-a)
end

function qc_delta(z, a, y)
    (a-y) .* sigmoid_prime(z)
end

function return_qc()
    QuadraticCost(qc_fn, qc_delta)
end

# Network struct and methods
mutable struct Network
    num_layers::Int
    sizes::Array
    biases::Array # of vectors
    weights::Array # of matrices
end

# methods on Network struct
function init(sizes::Vector; init_type="default")
    num_layers = length(sizes)
    weights, biases = weight_init(num_layers, sizes, init_type=init_type)
    Network(num_layers, sizes, biases, weights)
end

function weight_init(num_layers, sizes::Vector; init_type="default")
    biases = [randn(ns) for ns in sizes[2:end]]
    if init_type == "default"
        weights = [randn(sizes[l],sizes[l-1])/sizes[l-1] for l in 2:num_layers]
    else
        weights = [randn((sizes[ns],sizes[ns-1])) for ns in 2:num_layers]
    end
    weights, biases
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
            update_mini_batch!(net, mini_batch, learning_rate, n)
        end
        if n_test > 0
            acc = evaluate(net, test_data)
            println("Epoch $i: $acc")
        else
            println("Epoch $i")
        end
    end
end

function update_mini_batch!(net::Network, mini_batch, learning_rate, n; lambda=5.0)
    grad_b = [zeros(size(b)) for b in net.biases]
    grad_w = [zeros(size(w)) for w in net.weights]
    for (x,y) in mini_batch
        (delta_grad_b, delta_grad_w) = backprop(net, x,y)
        grad_b += delta_grad_b
        grad_w += delta_grad_w
    end
    mini_batch_size = length(mini_batch)
    net.biases -= (learning_rate/mini_batch_size) * grad_b
    net.weights *= (1 - ((lambda * learning_rate) / n))
    net.weights -= (learning_rate/mini_batch_size) * grad_w
end

function backprop(net::Network, x::Vector, y::Vector; cost="cross-entropy")
    grad_b = [zeros(size(b)) for b in net.biases]
    grad_w = [zeros(size(w)) for w in net.weights]
    # feedforward
    (a, as, zs) = feedforward(net, x, fpass=true)
    # backward pass
    if cost=="quadratic"
        cost = return_qc()
    elseif cost=="cross-entropy"
        cost = return_cec()
    end
    error = cost.delta(zs[end], a, y)
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
    SGD!(net, train_data, 30, 10, 0.1, test_data=test_data)
    net
end

function test_mnist_quick()
    train_data, validation_data, test_data = load_data()
    net = init([784,10])
    SGD!(net, train_data, 5, 10, 3.0, test_data=test_data)
    net
end

function test_mnist_old()
    train_data, validation_data, test_data = load_data()
    net = init([784,10], init_type="old")
    SGD!(net, train_data, 3, 10, 3.0, test_data=test_data)
    net
end
