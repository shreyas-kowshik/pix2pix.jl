"""
using CUDAnative
device!(2)

using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save,@load
using Flux:testmode!
using Distributions:Normal
"""

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

BATCH_SIZE = 4

function sampleA2B(X_A_test,gen;base_id="1")
    """
    Samples new images in domain B
    X_A_test : N x C x H x W array - Test images in domain A
    """
    # testmode!(gen)
    X_A_test = norm(X_A_test)
    X_B_generated = cpu(gen(X_A_test |> gpu)).data
    println(size(X_B_generated))
    println(minimum(X_B_generated))
    println(maximum(X_B_generated))

    # testmode!(gen,false)
    imgs = []
    s = size(X_B_generated)
    for i in 1:size(X_B_generated)[end]
       xt = reshape(X_A_test[:,:,:,i],256,256,3,1)
       xb = reshape(X_B_generated[:,:,:,i],256,256,3,1)
       out_array = cat(get_image_array(xt),get_image_array(xb),dims=3)
       save(string("./sample/",base_id,"_$i.png"),colorview(RGB,out_array))
    end
    imgs
end

function test()
   # load test data
   data = load_dataset("../data/edges2shoes/train/",256)[17343 + 4:17394 + 4]
   println(data[1])
   
   # Split into batches
   mb_idxs = partition(1:length(data), BATCH_SIZE)
   train_batches = [data[i] for i in mb_idxs]
    
   @load "../weights/e2s/gen_5000.bson" gen
    
   println(gen)

   # HACK ON GENERATOR #
   for block in gen.conv_down_blocks
    for layer in block.layers
     if typeof(layer) <: Dropout
      layer.p = 0.0
     end
    end
   end

   for block in gen.conv_blocks
    for layer in block.layers
     if typeof(layer) <: Dropout
      layer.p = 0.0
     end
    end
   end

   for (i,block) in enumerate(gen.up_blocks)
    if i == length(gen.up_blocks)
     break
    end
    
    for layer in block.upsample.layers
     if typeof(layer) <: Dropout
      layer.p = 0.0
     end
    end
   end
   #####################

   println(gen)

   gen = gen |> gpu
    
   println("Loaded Generator")

   for i in 1:length(train_batches)
        println(train_batches[i])
   	data_mb,_ = get_batch(train_batches[i],256) |> gpu
   	data_mb = reshape(data_mb,256,256,3,BATCH_SIZE) 
   	out = sampleA2B(data_mb,gen;base_id=string(i))
   end
end

# test()
