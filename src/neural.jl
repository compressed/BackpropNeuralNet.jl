type NeuralNetwork
  structure::Array{Int64, 1}
  disable_bias::Bool
  learning_rate::Float64
  momentum::Float64
  initial_weight_function::Function
  propagation_function::Function
  derivative_propagation_function::Function
  activation_nodes::Array{Any, 1}
  weights::Array{Any, 1}
  last_changes::Array{Any, 1}
  deltas::Array{Any, 1}
end

NeuralNetwork(structure::Array{Int64, 1}, disable_bias::Bool) = NeuralNetwork(
  structure,
  disable_bias,
  0.25,
  0.1,
  () -> randi(2000)/1000.0 - 1,
  (x::Float64) -> 1/(1+exp(-1*(x))),
  (y::Float64) -> y*(1-y),
  {},
  {},
  {},
  {}
)

function init_network(structure::Array{Int64,1})
  network = NeuralNetwork(structure, false)
  init_activation_nodes(network)
  init_weights(network)
  init_last_changes(network)
  return network
end

function init_activation_nodes(network::NeuralNetwork)
  network.activation_nodes = {}
  # for each layer in network, build 1.0 matrices
  for i in 1:length(network.structure)
    if !network.disable_bias && i < length(network.structure)
      push(network.activation_nodes, ones(network.structure[i] + 1))
    else
      push(network.activation_nodes, ones(network.structure[i]))
    end
  end
end

function init_weights(network::NeuralNetwork)
  network.weights = {}
  for i in 1:length(network.structure)-1
    arr = Array(Float64, length(network.activation_nodes[i]), network.structure[i+1])

    for j=1:numel(arr)
      arr[j] = network.initial_weight_function()
    end

    push(network.weights, arr)
  end
end

function init_last_changes(network::NeuralNetwork)
  network.last_changes = {}

  for i=network.weights
    push(network.last_changes, [zeros(size(i))])
  end
end

function train(network::NeuralNetwork, inputs::Vector{Float64}, outputs::Vector{Float64})
  net_eval(network, inputs)
  backpropagate(network, outputs)
  calculate_error(network, outputs)
end

function net_eval(network::NeuralNetwork, inputs::Vector{Float64})
  check_input_dimension(network, inputs)
  if length(network.weights) == 0
    init_network(network)
  end
  feedforward(network, inputs)
  return network.activation_nodes[end]
end

function feedforward(network::NeuralNetwork, inputs::Vector{Float64})
  for i in 1:length(inputs)
    network.activation_nodes[1][i] = inputs[i]
  end

  for n in 1:length(network.weights)
    for j in 1:network.structure[n+1]
      s = dot(network.activation_nodes[n], network.weights[n][:, j])
      network.activation_nodes[n+1][j] = network.propagation_function(s)
    end
  end
end

function backpropagate(network::NeuralNetwork, expected_values::Vector{Float64})
  check_output_dimension(network, expected_values)
  calculate_output_deltas(network, expected_values)
  calculate_internal_deltas(network)
  update_weights(network)
end

function calculate_output_deltas(network::NeuralNetwork, expected_values::Vector{Float64})
  output_values = network.activation_nodes[end]
  err = expected_values - output_values
  output_deltas = Array(Float64, 1, length(err))
  for i=1:length(err)
    output_deltas[i] = network.derivative_propagation_function(output_values[i]) * err[i]
  end
  network.deltas = {output_deltas}
end

function calculate_internal_deltas(network::NeuralNetwork)
  prev_deltas = network.deltas[end]
  for layer_index=2:length(network.activation_nodes)-1
    layer_deltas = Array(Float64,1,length(network.activation_nodes[layer_index]))
    for j=1:length(network.activation_nodes[layer_index])
      err = 0.0
      for k=1:network.structure[layer_index+1]
        err += prev_deltas[k] * network.weights[layer_index][j,k]
      end
      layer_deltas[j] = network.derivative_propagation_function(network.activation_nodes[layer_index][j]) * err
    end
    unshift(network.deltas, layer_deltas)
  end
end

function update_weights(network::NeuralNetwork)
  for n=1:length(network.weights)
    for i=1:size(network.weights[n],1)
      for j=:1:size(network.weights[n],2)
        change = network.deltas[n][j] * network.activation_nodes[n][i]
        network.weights[n][i,j] += (network.learning_rate * change + network.momentum * network.last_changes[n][i,j])
        network.last_changes[n][i,j] = change
      end
    end
  end
end

function calculate_error(network::NeuralNetwork, expected_output::Vector{Float64})
  output_values = network.activation_nodes[end]
  err = 0.0
  diff = output_values - expected_output
  for output_index=1:length(diff)
    err +=
      0.5 * diff[output_index]^2
  end
  return err
end

# TODO: throw exception here..
function check_input_dimension(network::NeuralNetwork, inputs::Vector{Float64})
  if length(inputs) != network.structure[1]
    error("Wrong number of inputs.\n",
          strcat("Expected: ", network.structure[1], "\n"),
          strcat("Received: ", length(inputs)))
  end
end

function check_output_dimension(network::NeuralNetwork, outputs::Vector{Float64})
  if length(outputs) != network.structure[end]
    error("Wrong number of outputs.\n",
          strcat("Expected: ", network.structure[end], "\n"),
          strcat("Received: ", length(outputs)))
  end
end
