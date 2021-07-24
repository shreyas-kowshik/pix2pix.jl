using Flux
using Statistics
using Distributions

function _random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

model = Chain(
    Conv((3, 3), 3=>16, pad = (1, 1); init=_random_normal),
    BatchNorm(16),
    x->leakyrelu.(x, 0.2)
)
