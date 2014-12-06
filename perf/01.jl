load("src/neural.jl")

# Force JIT'ing
net = init_network([2, 3, 2])
train(net, [0.15, 0.7],[0.1, 0.9])
net_eval(net, [0.15, 0.7])

@elapsed for i = 1:10_000
  net = init_network([2, 3, 2])
  train(net, [0.15, 0.7],[0.1, 0.9])
  net_eval(net, [0.15, 0.7])
end
