using Flux
using Statistics

model = Chain(
Dense(100, 1),
x->leakyrelu.(x, 0.2f0)
)

opt_gen = ADAM(0.001)

custom_loss(x) = mean(model(x) .- ones(size(model(x))...).^2)

X = randn(100, 1000)

# Explicit gradient computation #
# Does not seem that one needs to explicitly zero_grad! it #
# Check repeatedly #
ps = Flux.params(model)
# Taking gradient
loss, back = Flux.pullback(ps) do
    custom_loss(X)
end
grad = back(1f0)
# update!(opt, ps, grad)
