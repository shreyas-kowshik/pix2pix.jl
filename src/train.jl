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
NUM_EPOCHS = 50
BATCH_SIZE = 1
dis_lr = 0.000002f0
gen_lr = 0.0002f0
λ = 10.0 # L1 reconstruction Loss Weight
NUM_EXAMPLES = 1 # Temporary for experimentation
VERBOSE_FREQUENCY = 2 # Verbose output after every 2 epochs
# Debugging
G_STEPS = 10
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
dis = Discriminator() |> gpu # Discriminator
println("Loaded Models")

# Define Optimizers
# opt_gen = ADAM(gen_lr,(0.5,0.999))
# opt_disc = ADAM(dis_lr,(0.5,0.999))
opt_gen = ADAM(params(gen),gen_lr,β1=0.5)
opt_disc = ADAM(params(dis),dis_lr,β1=0.5)
# opt_disc = Descent(dis_lr)

# Forward prop, backprop, optimise!
function train_step(X_A,X_B) 
    # println("IN")
    # Normalise the Images
    # println(maximum(cpu(X_A)))
    # println(minimum(cpu(X_A)))
    # save_to_image(X_A)
    X_A = norm(X_A)
    X_B = norm(X_B)
    save_to_image(X_A,"A.png")
    save_to_image(X_B,"B.png")
    println("Saved image")

    loss_D = 0.0
    for _ in 1:D_STEPS
        # LABELS #
        # Flip the labels -> Not now, Noisy labels
        real_labels = rand(Uniform(0.9,1.0),1,BATCH_SIZE) |> gpu
        fake_labels = rand(Uniform(0,0.1),1,BATCH_SIZE) |> gpu
        # real_labels = ones(1,BATCH_SIZE) |> gpu
        # fake_labels = zeros(1,BATCH_SIZE) |> gpu
        
        ### Forward Propagation ###
        ### Discriminator Update ###
        zero_grad!(dis)

        # println("Before forward")
        # Domain A->B
        fake_B = gen(X_A)

        # println("After forward")
        save_to_image(fake_B.data,"t.png")
        fake_AB = cat(fake_B,X_A,dims=3) |> gpu
        fake_prob = drop_first_two(dis(fake_AB))
        loss_D_fake = bce(fake_prob,fake_labels)
        # loss_D_fake = logitbinarycrossentropy(fake_prob,fake_labels)
        println(fake_prob)
        println(fake_labels)

        real_AB =  cat(X_B,X_A,dims=3) |> gpu
        real_prob = drop_first_two(dis(real_AB))
        loss_D_real = bce(real_prob,real_labels)
        # loss_D_real = logitbinarycrossentropy(real_prob,real_labels)
        println(real_prob)
        println(real_labels)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        # println(loss_D)
        Flux.back!(loss_D)
        opt_disc()
    end

    # # Optimise #
    # gs = Tracker.gradient(() -> loss_D,params(dis))
    # # println(mean(gs[dis.layers[1].weight]))
    # update!(opt_disc,params(dis),gs) 
    
    loss_G = 0.0
    for _ in 1:G_STEPS
        ### Generator Update ###
        zero_grad!(gen)

        # No noisy labels for the generator
        real_labels = ones(1,BATCH_SIZE) |> gpu
        fake_labels = zeros(1,BATCH_SIZE) |> gpu

        # Domain A->B
        fake_B = gen(X_A)

        fake_AB = cat(fake_B,X_A,dims=3) |> gpu
        fake_prob = drop_first_two(dis(fake_AB))
        println("Fake_Prob Generator : $fake_prob")
        loss_adv = bce(fake_prob,real_labels)
        # loss_adv = logitbinarycrossentropy(fake_prob2,real_labels)
        
        loss_L1 = mean(abs.(fake_B .- X_B) |> gpu)

        println("Loss_L1 : $loss_L1")
        println("Loss adversarial : $loss_adv")

        loss_G = loss_adv + λ*loss_L1
        Flux.back!(loss_G)
        opt_gen()
    end

    # Optimise #
    # gs = Tracker.gradient(() -> loss_G,params(gen))  
    # update!(opt_gen,params(gen),gs)

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
            d_loss,g_loss = train_step(train_A[i] |> gpu,train_B[i] |> gpu)
            if epoch % VERBOSE_FREQUENCY == 0
                println("Gen Loss : $g_loss")
                println("DisA Loss : $d_loss")
            end
        end
    end

    save_weights(gen,dis)
end

train()