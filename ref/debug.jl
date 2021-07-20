using CUDAnative
device!(5)

using BenchmarkTools
using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using Distributions:Normal

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

dis = Discriminator() |> gpu
gen = UNet() |> gpu
println(length(Flux.params(gen)))

function loss(a,b)
 fake = gen(a)
 t = cat(fake,b,dims=3)
 prob = dis(fake)
end

x = ones(256,256,3,1) |> gpu
y = ones(256,256,3,1) |> gpu

gs = Tracker.gradient(() -> loss(x,y),Flux.params(gen))
println(gs[dis.layers[end].weight])
"""
for i in 1:5
	println(i)
	@time loss(x)
	@time Tracker.gradient(() -> loss(x),Flux.params(gen))
end
"""

"""

m = Chain(Conv((3,3),3=>64,pad=(1,1))) |> gpu
x = rand(256,256,3,1) |> gpu

function loss(x)
   out = m(x)
   mean(out)
end

for _ in 1:5
	@time o = loss(x)
	@time gs = Tracker.gradient(() -> loss(x),Flux.params(m))
end
"""
