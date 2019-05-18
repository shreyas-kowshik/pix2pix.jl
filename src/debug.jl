using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

# dis = Discriminator() |> gpu
gen = UNet() |> gpu
fake_B = gen(ones(256,256,3,1) |> gpu)
# fake_AB = cat(fake_B,ones(256,256,3,1) |> gpu,dims=3)
# fake_prob = drop_first_two(dis(fake_AB))
# loss = mean((fake_prob .- ones(size(fake_prob)))
loss = mean(abs.(fake_B .- rand(size(fake_B))))
# loss2 = mean()

gs = Tracker.gradient(() -> loss,params(gen))
# update!(opt_gen,params(gen),gs)

# out = dis(fake_AB.data)
# println(size(out))

# out = drop_first_two(out)
# println(size(out))

# m = Chain(Conv((3,3), 3=>1,pad = (1, 1)),MaxPool((2,2)),ConvTranspose((2, 2), 1=>1, stride=(2, 2)),MaxPool((2,2))) |> gpu
# out = m(ones(4,4,3,1) |> gpu)
# println(size(out))
# loss = mean(abs.(out .- (rand(size(out)) |> gpu)))

# gs = Tracker.gradient(() -> loss,params(m))