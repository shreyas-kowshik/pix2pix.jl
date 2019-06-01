using CUDAnative
device!(4)

using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save
using Flux:testmode!
using Distributions:Normal,Uniform

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

# Hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 4
dis_lr = 0.00002f0
gen_lr = 0.00002f0
λ = convert(Float32,10.0) # L1 reconstruction Loss Weight
NUM_EXAMPLES = 1 # Temporary for experimentation
VERBOSE_FREQUENCY = 10 # Verbose output after every 10 steps
SAVE_FREQUENCY = 2000
# Debugging
G_STEPS = 1
D_STEPS = 1

# Global printing variables
global gloss = 0.0
global dloss = 0.0

# Data Loading
data = load_dataset("../data/edges2shoes/train/",256)

mb_idxs = partition(shuffle!(collect(1:length(data))), BATCH_SIZE)
train_batches = [data[i] for i in mb_idxs]
println("Loaded Data")

# Define models
gen = UNet() |> gpu # Generator For A->B
dis = Discriminator() |> gpu
println("Loaded Models")

# Define Optimizers
opt_gen = ADAM(gen_lr,(0.5,0.999))
opt_disc = ADAM(dis_lr,(0.5,0.999))

function d_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    global dloss
    real_labels = ones(1,size(a)[end]) |> gpu
    fake_labels = zeros(1,size(a)[end]) |> gpu
    
    fake_B = gen(a |> gpu)
    fake_AB = cat(fake_B,a,dims=3) |> gpu

    fake_prob = drop_first_two(dis(fake_AB))
    loss_D_fake = bce(fake_prob,fake_labels)

    real_AB =  cat(b,a,dims=3) |> gpu
    real_prob = drop_first_two(dis(real_AB))
    loss_D_real = bce(real_prob,real_labels)

    dloss = convert(Float32,0.5) * mean(loss_D_real .+ loss_D_fake)
    dloss
end

function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    global gloss
    # println(mean(b))
    real_labels = ones(1,size(a)[end]) |> gpu
    fake_labels = zeros(1,size(a)[end]) |> gpu
    
    fake_B = gen(a |> gpu)
    fake_AB = cat(fake_B,a,dims=3) |> gpu

    fake_prob = drop_first_two(dis(fake_AB))
    
    # println("Fake Prob : $fake_prob")
    loss_adv = mean(bce(fake_prob,real_labels))
    loss_L1 = mean(abs.(fake_B .- b)) 
    # println("Loss L1 : $loss_L1")
    gloss = loss_adv + λ*loss_L1
    gloss
end

# Forward prop, backprop, optimise!
function train_step(X_A,X_B)
    global gloss
    global dloss
    start = time()
    X_A = norm(X_A)
    X_B = norm(X_B)
    time_ = time() - start
    println("Normalizations : $time_")

    start = time()
    gs = Tracker.gradient(() -> d_loss(X_A,X_B),params(dis))
    time_ = time() - start
    println("Dis gradient : $time_")

    start = time()
    update!(opt_disc,params(dis),gs)
    time_ = time() - start
    println("Dis update : $time_")

    start = time()
    gs = Tracker.gradient(() -> g_loss(X_A,X_B),params(gen))  
    time_ = time() - start
    println("Gen gradient : $time_")

    start = time()
    update!(opt_gen,params(gen),gs)
    time_ = time() - start
    println("Gen update : $time_")

    # Get losses
    # loss_G = g_loss(X_A,X_B)
    # loss_D = d_loss(X_A,X_B)

    # return loss_D,loss_G
end

function save_weights(gen,dis)
    gen = gen |> cpu
    dis = dis |> cpu
    @save "../weights/e2s/gen.bson" gen
    @save "../weights/e2s/dis.bson" dis
    gen = gen |> gpu
    dis = dis |> gpu
    println("Saved...")
end

function train()
    global gloss
    global dloss

    println("Training...")
    verbose_step = 0

    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
        for i in 1:length(train_batches)
	    glob_start = time()
	    start = time()
	    train_A,train_B = get_batch(train_batches[i],256)
            time_ = time() - start
	    println("get_batch : $time_")
	    # println(mean(train_B))
	    st = time()
            train_step(train_A |> gpu,train_B |> gpu)
	    time_ = time() - st
	    println("Train step : $time_")
            if verbose_step % VERBOSE_FREQUENCY == 0
		println("--- Verbose Step : $verbose_step ---")
                println("Gen Loss : $gloss")
                println("Dis Loss : $dloss")
            end
	    
	    if verbose_step % SAVE_FREQUENCY == 0
		start = time()
		save_weights(gen,dis)
		time_ = time() - start
		println("Save : $time_")
	    end

	    verbose_step+=1    
	    time_ = time() - glob_start
	    println("")
	    println("TRAIN BATCH : $time_")
	    println("-------------------------")
        end
    end

    save_weights(gen,dis)
end

train()
