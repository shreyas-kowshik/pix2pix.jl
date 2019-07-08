using CUDAnative
device!(1)

using Flux,CuArrays
using Flux:Tracker
using Flux:@treelike
using Statistics
using BenchmarkTools
using TimerOutputs

const to = TimerOutput()

c() = Chain(Conv((3,3),1=>64),Conv((3,3),64=>256),Conv((3,3),256=>512))

struct Net
    u
    v
end

@treelike Net

function Net()
   u = Chain(Conv((3,3),3=>64))
   v = Chain(c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c(),c())
   Net(u,v)
end

function (n::Net)(x)
   return n.u(x)
end

m = Net() |> gpu
x = rand(256,256,3,1) |> gpu

function loss(x)
   out = m(x)
   mean(out)
end

function test()
    @timeit to "loss" loss(x)
    @timeit to "grads" Tracker.gradient(() -> loss(x),Flux.params(m))
end

@timeit to "loss" loss(x)
@timeit to "grads" Tracker.gradient(() -> loss(x),Flux.params(m))

for _ in 1:5
	@btime loss(x)
	@btime Tracker.gradient(() -> loss(x),Flux.params(m))
end

println(to)
"""
for _ in 1:5
	test()
end
"""
