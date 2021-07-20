using CUDAnative
using CUDAnative:exp,log
device!(3)
println("Device Selected")

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
NUM_EPOCHS = 2000
BATCH_SIZE = 4
dis_lr = 0.0002f0
gen_lr = 0.0002f0
λ = 100.0f0 # L1 reconstruction Loss Weight
NUM_EXAMPLES = 10  # Temporary for experimentation
VERBOSE_FREQUENCY = 1 # Verbose output after every 10 steps
SAVE_FREQUENCY = 500
SAMPLE_FREQUENCY = 5 # Sample every these mamy number of steps
# Debugging
G_STEPS = 1 
D_STEPS = 1
resume = false
inverse_order = true

# Global printing variables
global gloss = 0.0
global dloss = 0.0

# Statistics to keep track of
global gloss_hist = []
global dloss_hist = []
global global_step = 0

# Data Loading
data = load_dataset("../../old_server_pix2pix/data/facades/train/",256)[1:NUM_EXAMPLES]
println(data[1])
println(length(data))

mb_idxs = partition(shuffle!(collect(1:length(data))), BATCH_SIZE)
train_batches = [data[i] for i in mb_idxs]
println(length(train_batches))
println("Loaded Data")

# Define Optimizers
# opt_gen = ADAM(gen_lr,(0.5,0.999))
# opt_disc = ADAM(dis_lr,(0.5,0.999))

function d_loss(gen,dis,a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    global dloss
    
    fake_B = gen(a).data
    save_to_image(fake_B,"tem.jpg")
    save_to_image(b,"out.jpg")
    save_to_image(a,"a.jpg")

    fake_AB = cat(fake_B,a,dims=3)

    fake_prob = dis(fake_AB)
    println("Fake prob done ")
    println("Size Dis : $(size(fake_prob))")
    
    # println("-------------DIS----------")
    # println("Fake prob : $fake_prob")
    
    fake_labels = param(zeros(size(fake_prob)...)) |> gpu
    loss_D_fake = logitbinarycrossentropy(fake_prob,fake_labels)
    # loss_D_fake = mean(bce(fake_prob,fake_labels))

    # println("Fake Loss : $(mean(loss_D_fake))")

    real_AB =  cat(b,a,dims=3)
    real_prob = dis(real_AB)
    # println("Real Prob : $real_prob")
    real_labels = param(ones(size(real_prob)...)) |> gpu

    loss_D_real = logitbinarycrossentropy(real_prob,real_labels)
    # loss_D_real = mean(bce(real_prob,real_labels))
    # println("Real Loss : $(mean(loss_D_real))")

    println(mean(fake_prob))
    println(mean(real_prob))
    
    dloss = 0.5 * mean(loss_D_real.data .+ loss_D_fake.data)
    # convert(Float32,0.5) * mean(loss_D_real .+ loss_D_fake)
    return loss_D_real + loss_D_fake
    # retrun loss
end

function g_loss(gen,dis,a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    global gloss
    
    fake_B = gen(a)
    save_to_image(fake_B.data,"tem1.jpg")
    save_to_image(b,"out1.jpg")
    save_to_image(a,"a1.jpg")

    fake_AB = cat(fake_B,a,dims=3)

    start = time()
    fake_prob = dis(fake_AB)

    println("Gen prob : $(mean(fake_prob))")
     
    time_ = time() - start
    # println("fake_prob : $time_")
    
    real_labels = param(ones(size(fake_prob)...)) |> gpu
    
    # println("---------------------GEN-----------------")
    # println("Fake Prob : $fake_prob")
    loss_adv = mean(logitbinarycrossentropy(fake_prob,real_labels))
    # loss_adv = mean(bce(fake_prob,real_labels))
    println("Loss Adv : $loss_adv")

    loss_L1 = mean(abs.(fake_B .- b))
    println("Loss L1 : $loss_L1")

    gloss = loss_adv.data + λ*loss_L1.data
    return loss_adv # + 10.0f0 * loss_L1
end

# Forward prop, backprop, optimise!
function train_step(gen,dis,X_A,X_B,opt_gen,opt_disc)
    global gloss
    global dloss
    start = time()
    println(minimum(X_B))
    println(maximum(X_B))
    X_A = norm(X_A)
    X_B = norm(X_B)
    println(minimum(X_B))
    println(maximum(X_B))

    for _ in 1:D_STEPS
	   # zero_grad!(gen)
           zero_grad!(dis)
	
	   Flux.back!(d_loss(gen,dis,X_A,X_B))
	   opt_disc()

   	   # gs = Tracker.gradient(() -> d_loss(gen,dis,X_A,X_B),params(dis))
	   # println("DIs Bottom Grad : $(mean(gs[dis.layers[1].weight]))")
	   # println("Dis Top Grad : $(mean(gs[dis.layers[end-1].weight]))") 
	   # println("Dis Top Grad : $(mean(gs[dis.layers[end].layers[end-2].weight]))")

       	   # update!(opt_disc,params(dis),gs)
	   # zero_grad!(gen)
           # zero_grad!(dis)

	   # println("After Update")
	   # println("DIs Bottom Grad : $(mean(gs[dis.layers[1].weight]))")
	   # println("Dis Top Grad : $(mean(gs[dis.layers[end-1].weight]))")
           # println("Dis Top Grad : $(mean(gs[dis.layers[end].layers[end-2].weight]))")
    end

    for _ in 1:G_STEPS
    zero_grad!(gen)
    # zero_grad!(dis)
    Flux.back!(g_loss(gen,dis,X_A,X_B))
    opt_gen()

    # gs = Tracker.gradient(() -> g_loss(gen,dis,X_A,X_B),params(gen))  

    # println("\n\n\n\n\n\n")
    # println("Gen bottom grad : $(mean(gs[gen.conv_blocks[1].layers[1].weight]))")
    #  println("Gen top grad : $(mean(gs[gen.up_blocks[end].layers[end].weight]))")
    
    # update!(opt_gen,params(gen),gs)
    # zero_grad!(gen)
    # zero_grad!(dis)

    # println("After Update")
    # println("Gen bottom grad : $(mean(gs[gen.conv_blocks[1].layers[1].weight]))")
    # println("Gen top grad : $(mean(gs[gen.up_blocks[end].layers[end].weight]))")
    end
end

function save_weights(gen,dis)
    gen = gen |> cpu
    dis = dis |> cpu
    @save "../weights/facades/gen.bson" gen
    @save "../weights/facades/dis.bson" dis
    gen = gen |> gpu
    dis = dis |> gpu
    println("Saved...")
end

function train()
    global gloss
    global dloss
    global global_step

    println("Training...")
    verbose_step = 0

    # Define models
    if resume == true
	@load "../weights/facades/gen.bson" gen
	@load "../weights/facades/dis.bson" dis
	gen = gen |> gpu
	dis = dis |> gpu
	println("Loaded Networks")
    else
    	gen = UNet() |> gpu # Generator For A->B
    	dis = Discriminator() |> gpu
	println("Initialized Neworks")
    end
    println(length(params(gen)))
    println(length(params(dis)))
    println("Loaded Models")


    opt_gen = ADAM(params(gen),gen_lr,β1 = 0.5)
    opt_disc = ADAM(params(dis),dis_lr,β1 = 0.5)

    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
	
	mb_idxs = partition(shuffle!(collect(1:length(data))), BATCH_SIZE)
	train_batches = [data[i] for i in mb_idxs]
	
        for i in 1:length(train_batches)
	    global_step += 1
	    
	    if global_step % 7000 == 0
		# opt_gen.eta = opt_gen.eta / 1.0
		# opt_disc.eta = opt_disc.eta / 1.0
	    end
	      
	    glob_start = time()
	    start = time()
	
	    print("Train batches ; ")
	    println(length(train_batches[i]))
	    if inverse_order == false
	    	train_A,train_B = get_batch(train_batches[i],256)
	    else
		train_B,train_A = get_batch(train_batches[i],256)
	    end
	    
	    println(size(train_A))
            time_ = time() - start
#	    println("get_batch : $time_")
	    # println(mean(train_B))
	    st = time()
            train_step(gen,dis,train_A |> gpu,train_B |> gpu,opt_gen,opt_disc)
	    time_ = time() - st
	    # println("Train step : $time_")

	    # push!(dloss_hist,dloss)
	    # push!(gloss_hist,gloss)
	    
            if verbose_step % VERBOSE_FREQUENCY == 0
		i = plot(dloss_hist,fmt=:png)
		j = plot(gloss_hist,fmt=:png)

		# Save the statistics
		# save("../weights/stats.jld","dloss",dloss_hist)
		# save("../weights/stats.jld","dloss",dloss_hist)

		println("--- Verbose Step : $verbose_step ---")
                println("Gen Loss : $gloss")
                println("Dis Loss : $dloss")
            end
	
	    if verbose_step % (100 * SAVE_FREQUENCY) == 0
	    	# savefig(i,"dloss.png")
	    	# savefig(j,"gloss.png")
		println("")
	    end
	    
	    if verbose_step % SAMPLE_FREQUENCY == 0	
                sampleA2B(train_A,gen)
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
