using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics
using Flux.Tracker:update!
using BSON: @save
using Flux:testmode!
using Distributions:Normal

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 1
dis_lr = 0.0002f0
gen_lr = 0.0002f0
λ = 100.0 # L1 reconstruction Loss Weight
NUM_EXAMPLES = 2 # Temporary for experimentation
VERBOSE_FREQUENCY = 2 # Verbose output after every 2 epochs

# Data Loading
dataA,dataB = load_dataset("../data/train/",256)

# Temporary
dataA = dataA[:,:,:,1:NUM_EXAMPLES]
dataB = dataB[:,:,:,1:NUM_EXAMPLES]
###########

mb_idxs = partition(shuffle!(collect(1:size(dataA)[end])), BATCH_SIZE)
train_A = [make_minibatch(dataA, i) for i in mb_idxs]
train_B = [make_minibatch(dataB, i) for i in mb_idxs]
println("Loaded Data")

# Define Optimizers
opt_gen = ADAM(gen_lr,(0.5,0.999))
opt_disc = ADAM(dis_lr,(0.5,0.999))
# opt_disc = Descent(dis_lr)

# Define models
gen = UNet() |> gpu # Generator For A->B
dis = Discriminator() |> gpu # Discriminator
println("Loaded Models")

# Forward prop, backprop, optimise!
function train_step(X_A,X_B) 
    # Normalise the Images
    X_A = norm(X_A)
    X_B = norm(X_B)

    # LABELS #
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu
    
    ### Forward Propagation ###
    # Domain A->B
    fake_B = gen(X_A)

    ### Discriminator Update ###
    fake_AB = cat(fake_B,X_A,dims=3) |> gpu
    fake_prob = drop_first_two(dis(fake_AB))
    loss_D_fake = bce(fake_prob,fake_labels)
    # println(fake_prob)
    # println(fake_labels)

    real_AB =  cat(X_B,X_A,dims=3) |> gpu
    real_prob = drop_first_two(dis(real_AB))
    loss_D_real = bce(real_prob,real_labels)
    # println(real_prob)
    # println(real_labels)
    
    loss_D = 0.5 * (loss_D_real + loss_D_fake)
    
    ### Generator Update ###
    fake_AB2 = cat(fake_B,X_A,dims=3) |> gpu
    fake_prob2 = drop_first_two(dis(fake_AB2))
    loss_adv = bce(fake_prob2,real_labels)

    loss_L1 = mean(abs.(fake_B .- X_B) |> gpu)

    loss_G = loss_adv + λ*loss_L1

    # Optimise #
    gs = Tracker.gradient(() -> loss_D,params(dis))
    update!(opt_disc,params(dis),gs)  

    gs = Tracker.gradient(() -> loss_G,params(gen))  
    update!(opt_gen,params(gen),gs)

    return loss_D,loss_G
end

function save_weights()
    gen_weights = Tracker.data.(params(cpu(gen)))
    dis_weights = Tracker.data.(params(cpu(dis)))
    @save "../weights/gen.bson" gen_weights
    @save "../weights/dis.bson" dis_weights
end

function train()
    println("Training...")

    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
        for i in 1:length(train_A)
            d_loss,g_loss = train_step(train_A[i] |> gpu,train_B[i] |> gpu)
            if epoch % VERBOSE_FREQUENCY == 0
                println("Gen Loss : $g_loss")
                println("DisA Loss : $d_loss")
            end
        end
    end

    save_weights()
end

train()