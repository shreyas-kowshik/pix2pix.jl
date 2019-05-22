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
    # testmode!(gen)
    X_A_test = norm(X_A_test)
    X_B_generated = cpu(denorm(gen(X_A_test |> gpu)).data)
    testmode!(gen,false)
    imgs = []
    s = size(X_B_generated)
    for i in size(X_B_generated)[end]
       push!(imgs,colorview(RGB,reshape(X_B_generated[:,:,:,i],3,s[1],s[2])))
    end
    imgs
end

function test()
   # gen = UNet()
   # println("Loaded Generator")

   # load test data
   dataA,_ = load_dataset("../data/train/",256)
   dataA = dataA[:,:,:,1] |> gpu
   dataA = reshape(dataA,256,256,3,1)

   # gen_weights = Tracker.data.(params(gen));

   # println(gen_weights)
   # @load "../weights/gen.bson" gen
   @load "../weights/gen.bson" gen

   # Flux.loadparams!(gen,gen_weights)
   gen = gen |> gpu
   println("Loaded Generator")
   
   out = sampleA2B(dataA,gen)

   for (i,img) in enumerate(out)
        save("../sample/A_$i.png",img)
   end
end

test()