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
NUM_EPOCHS = 5000
BATCH_SIZE = 1
dis_lr = 0.000002f0
gen_lr = 0.02f0
λ = convert(Float32,10.0) # L1 reconstruction Loss Weight
NUM_EXAMPLES = 1 # Temporary for experimentation
VERBOSE_FREQUENCY = 1 # Verbose output after every 2 epochs
# Debugging
G_STEPS = 1
D_STEPS = 1

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
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    fake_B = gen(a |> gpu)
    fake_AB = cat(fake_B,a,dims=3) |> gpu

    fake_prob = drop_first_two(dis(fake_AB))
    loss_D_fake = bce(fake_prob,fake_labels)

    real_AB =  cat(b,a,dims=3) |> gpu
    real_prob = drop_first_two(dis(real_AB))
    loss_D_real = bce(real_prob,real_labels)

    0.5 * mean(loss_D_real .+ loss_D_fake)
end

function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    fake_B = gen(a |> gpu)
    fake_AB = cat(fake_B,a,dims=3) |> gpu

    fake_prob = drop_first_two(dis(fake_AB))

    loss_adv = mean(bce(fake_prob,real_labels))
    loss_L1 = mean(abs.(gen(a |> gpu) .- (b |> gpu))) 
    loss_adv + λ*loss_L1
end

# Forward prop, backprop, optimise!
function train_step(X_A,X_B)
    X_A = norm(X_A)
    X_B = norm(X_B)

    loss_D = 0.0
    for _ in 1:D_STEPS
        gs = Tracker.gradient(() -> d_loss(X_A,X_B),params(dis))
        update!(opt_disc,params(dis),gs)
    end

    loss_G = 0.0
    for _ in 1:G_STEPS
        gs = Tracker.gradient(() -> g_loss(X_A,X_B),params(gen))  
        update!(opt_gen,params(gen),gs)
    end

    # Get losses
    loss_G = g_loss(X_A,X_B)
    loss_D = d_loss(X_A,X_B)

    return loss_D,loss_G
end

function save_weights(gen,dis)
    gen = gen |> cpu
    dis = dis |> cpu
    @save "../weights/gen.bson" gen
    @save "../weights/dis.bson" dis
end

function train()
    println("Training...")

    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
        for i in 1:length(train_A)
            dloss,gloss = train_step(train_A[i] |> gpu,train_B[i] |> gpu)
            if epoch % VERBOSE_FREQUENCY == 0
                println("Gen Loss : $gloss")
                println("DisA Loss : $dloss")
            end
        end
    end

    save_weights(gen,dis)
end

train()
