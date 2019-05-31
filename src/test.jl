using CUDAnative
device!(5)

using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save,@load
using Flux:testmode!
using Distributions:Normal

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

function sampleA2B(X_A_test,gen)
    """
    Samples new images in domain B
    X_A_test : N x C x H x W array - Test images in domain A
    """
    testmode!(gen)
    X_A_test = norm(X_A_test)
    X_B_generated = cpu(gen(X_A_test |> gpu))
    println(size(X_B_generated))
    testmode!(gen,false)
    imgs = []
    s = size(X_B_generated)
    for i in 1:size(X_B_generated)[end]
       xt = reshape(X_A_test[:,:,:,i],256,256,3,1)
       xb = reshape(X_B_generated[:,:,:,i],256,256,3,1)
       out_array = cat(get_image_array(xt),get_image_array(xb),dims=3)
       save("../sample/$i.png",colorview(RGB,out_array))
    end
    imgs
end

function test()
   # load test data
   data = load_dataset("../data/edges2shoes/train/",256)[271:280]
   dataA,_ = get_batch(data,256) |> gpu
   dataA = reshape(dataA,256,256,3,10)

   @load "../weights/e2s/gen.bson" gen

   gen = gen |> gpu
   println("Loaded Generator")
   
   out = sampleA2B(dataA,gen)
end

test()
