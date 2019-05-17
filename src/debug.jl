using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

# dis = Discriminator() |> gpu
# gen = UNet() |> gpu
# fake_AB = gen((cat(ones(256,256,3,1),ones(256,256,3,1),dims=3)) |> gpu)

# out = dis(fake_AB.data)
# println(size(out))

# out = drop_first_two(out)
# println(size(out))

a = zeros(1,1) |> gpu
b = ones(1,1) |> gpu
b[1] = 0.7
a[1] = 0.0
println(bce(b,a))
