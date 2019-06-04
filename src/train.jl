using CUDAnative
using CUDAnative:exp,log
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
using JLD
using Plots
# using BenchmarkTools

include("utils.jl")
include("generator.jl")
include("discriminator.jl")
include("test.jl")

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 1
dis_lr = 0.0002f0
gen_lr = 0.0002f0
λ = convert(Float32,10.0) # L1 reconstruction Loss Weight
NUM_EXAMPLES = 1  # Temporary for experimentation
VERBOSE_FREQUENCY = 10 # Verbose output after every 10 steps
SAVE_FREQUENCY = 2000
SAMPLE_FREQUENCY = 50 # Sample every these mamy number of steps
# Debugging
G_STEPS = 1
D_STEPS = 1

# Global printing variables
global gloss = 0.0
global dloss = 0.0

# Statistics to keep track of
global gloss_hist = []
global dloss_hist = []

# Data Loading
data = load_dataset("../data/edges2shoes/train/",256)[1:NUM_EXAMPLES]
println(length(data))

mb_idxs = partition(shuffle!(collect(1:length(data))), BATCH_SIZE)
train_batches = [data[i] for i in mb_idxs]
println(length(train_batches))
println("Loaded Data")

# Define models
gen = UNet() |> gpu # Generator For A->B
dis = Discriminator() |> gpu
println(length(params(gen)))
println(length(params(dis)))
println("Loaded Models")

# Define Optimizers
opt_gen = ADAM(gen_lr,(0.5,0.999))
opt_disc = ADAM(dis_lr,(0.5,0.999))
#opt_gen = ADAM(params(gen),gen_lr,β1 = 0.5)
#opt_disc = ADAM(params(dis),dis_lr,β1 = 0.5)

function d_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    global dloss
    real_labels = ones(1,size(a)[end]) |> gpu
    fake_labels = zeros(1,size(a)[end]) |> gpu
    
    fake_B = gen(a).data
#    println(mean(fake_B))
#    println(minimum(fake_B))
#    println(maximum(fake_B))

    fake_AB = cat(fake_B,a,dims=3)

    fake_prob = drop_first_two(dis(fake_AB))
    println("-------------DIS----------")
    println("Fake prob : $fake_prob")
    
    # Intermediate calucaltion for gradient computation #
    o = mean(Chain(dis.layers[1:end-2]...)(fake_AB))
    println("Intermediate gradient_ value fake : $o")
    #############################

    loss_D_fake = bce(fake_prob,fake_labels)
    println("Fake Loss : $(mean(loss_D_fake))")

    real_AB =  cat(b,a,dims=3)
    real_prob = drop_first_two(dis(real_AB))
    println("Real Prob : $real_prob")

    # Intermediate calucaltion for gradient computation #
    o = mean(Chain(dis.layers[1:end-2]...)(real_AB))
    println("Intermediate gradient_ value real : $o")
    #############################

    loss_D_real = bce(real_prob,real_labels)
    println("Real Loss : $(mean(loss_D_real))")

    dloss = convert(Float32,0.5) * mean(loss_D_real .+ loss_D_fake)
    dloss
end

function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    global gloss
    global gen
    global dis

    glob_start = time()
    # println(mean(b))
    real_labels = ones(1,size(a)[end]) |> gpu
    fake_labels = zeros(1,size(a)[end]) |> gpu
    
    start = time()
    fake_B = gen(a)
    time_ = time() - start
    # println("fake_B : $time_")

    fake_AB = cat(fake_B,a,dims=3)

    start = time()
    fake_prob = drop_first_two(dis(fake_AB))
    time_ = time() - start
    # println("fake_prob : $time_")
    
    println("---------------------GEN-----------------")
    println("Fake Prob : $fake_prob")
    loss_adv = mean(bce(fake_prob,real_labels))
    println("Loss Adv : $loss_adv")

    loss_L1 = mean(abs.(fake_B .- b)) 
    println("Loss L1 : $loss_L1")
    time_ = time() - glob_start
    println("Overall g_loss : $time_")

    gloss = loss_adv + λ*loss_L1
    gloss
end

# Forward prop, backprop, optimise!
function train_step(X_A,X_B)
    global gen
    global dis
    global gloss
    global dloss
    start = time()
    X_A = norm(X_A)
    X_B = norm(X_B)
    time_ = time() - start
#    println("Normalizations : $time_")

    for _ in 1:D_STEPS
	   start = time()
	   zero_grad!(dis)
	   #Flux.back!(d_loss(X_A,X_B))
	   #println("DIs Bottom Grad : $(mean(dis.layers[1].weight.grad))")
	   #println("Dis Top Grad : $(mean(dis.layers[end-1].weight.grad))")

   	   gs = Tracker.gradient(() -> d_loss(X_A,X_B),params(dis))
	   println("DIs Bottom Grad : $(mean(gs[dis.layers[1].weight]))")
	   println("Dis Top Grad : $(mean(gs[dis.layers[end-1].weight]))") 

	   time_ = time() - start
	   println("Dis gradient : $time_")

    	start = time()
    	update!(opt_disc,params(dis),gs)
	#opt_disc()
    	time_ = time() - start
  	println("Dis update : $time_")
    end

    start = time()
    zero_grad!(gen)
    #Flux.back!(g_loss(X_A,X_B))
    #println("Gen bottom grad : $(mean(gen.conv_blocks[1].layers[1].weight.grad))")
    #println("Gen top grad : $(mean(gen.up_blocks[end].weight.grad))")

    gs = Tracker.gradient(() -> g_loss(X_A,X_B),params(gen))  
    println("Gen bottom grad : $(mean(gs[gen.conv_blocks[1].layers[1].weight]))")
    println("Gen top grad : $(mean(gs[gen.up_blocks[end].weight]))")

    time_ = time() - start
    println("Gen gradient : $time_")

    start = time()
    update!(opt_gen,params(gen),gs)
    #opt_gen()

    time_ = time() - start
    println("Gen update : $time_")
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
    global gen
    global dis

    println("Training...")
    verbose_step = 0

    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
        for i in 1:length(train_batches)
	    glob_start = time()
	    start = time()
	    train_A,train_B = get_batch(train_batches[i],256)
	    println(size(train_A))
            time_ = time() - start
#	    println("get_batch : $time_")
	    # println(mean(train_B))
	    st = time()
            train_step(train_A |> gpu,train_B |> gpu)
	    time_ = time() - st
	    # println("Train step : $time_")

	    push!(dloss_hist,dloss.data)
	    push!(gloss_hist,gloss.data)

            if verbose_step % VERBOSE_FREQUENCY == 0
		i = plot(dloss_hist,fmt=:png)
		j = plot(gloss_hist,fmt=:png)
		# savefig(i,"dloss.png")
		# savefig(j,"gloss.png")

		# Save the statistics
		save("../weights/stats.jld","dloss",dloss_hist)
		save("../weights/stats.jld","dloss",dloss_hist)

		println("--- Verbose Step : $verbose_step ---")
                println("Gen Loss : $gloss")
                println("Dis Loss : $dloss")
            end

	    if verbose_step % SAMPLE_FREQUENCY == 0	
                sampleA2B(train_A,gen)
	    end
	    
	    if verbose_step % SAVE_FREQUENCY == 0
		start = time()
		# save_weights(gen,dis)
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

    # save_weights(gen,dis)
end

train()
