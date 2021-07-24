# using CUDAnative
# using CUDAnative:exp,log
# using Images,CuArrays,Flux
# using Flux:@treelike, Tracker
# using Base.Iterators: partition
# using Random
# using Statistics
# using Flux.Tracker:update!
# using BSON: @save
# using Flux:testmode!
# using Distributions:Normal,Uniform
# using JLD
# using Plots
# # using BenchmarkTools

# Hyperparameters
# device=cpu
# NUM_EPOCHS = 2000
# BATCH_SIZE = 4
# dis_lr = 0.0002f0
# gen_lr = 0.0002f0
# λ = 100.0f0 # L1 reconstruction Loss Weight
# NUM_EXAMPLES = 10  # Temporary for experimentation
# VERBOSE_FREQUENCY = 1 # Verbose output after every 10 steps
# SAVE_FREQUENCY = 500
# SAMPLE_FREQUENCY = 5 # Sample every these mamy number of steps
# DATASET_PATH = "facades/train/"
# IMG_SIZE=256

using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy


function discriminator_loss(real_logits, fake_logits, device)
    real_labels = param(ones(size(real_logits)...)) |> device
    real_loss = logitbinarycrossentropy.(real_logits, real_labels)

    fake_labels = param(zeros(size(fake_logits)...)) |> device
    fake_loss = logitbinarycrossentropy.(fake_logits, fake_labels)

    0.5 * mean(real_loss + fake_loss)
end

function generator_loss(fake_output, fake_logits, domain_images, device; λ = 100.0f0)
    labels = param(ones(size(fake_logits)...)) |> device
    adv_loss = logitbinarycrossentropy.(fake_logits, real_labels)
    l1_loss = abs.(fake_output - domain_images)

    mean(adv_loss + (λ .* l1_loss))
end

function train_discriminator!(x, y, gen, discr, opt, device)
    """
    x : From Input Domain
    y : From Target Domain
    """
    fake_input = gen(x)
    disc_fake_input = cat(fake_input, x, dims=3)
    disc_real_input = cat(y, x, dims=3)
    fake_logits = discr(disc_fake_input)
    real_logits = discr(disc_real_input)
    
    ps = Flux.params(discr)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        discriminator_loss(fake_logits, real_logits, device))
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    return loss
end

function train_generator!(x, y, gen, discr, opt, device)
    """
    x : From Input Domain
    y : From Target Domain
    """
    fake_output = gen(x)
    disc_fake_input = cat(fake_output, x, dims=3)
    fake_logits = discr(disc_fake_input)
    
    ps = Flux.params(gen)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        generator_loss(fake_output, fake_logits, y, device)
    end
    grad = back(1f0)
    update!(opt, ps, grad)
    return loss
end

function train_step(x, y, gen, discr, opt_gen, opt_discr, device; D_STEPS=1, G_STEPS=1)
    """
    Optmize one step
    x : From Input Domain
    y : From Target Domain
    """
    x = norm(x)
    y = norm(y)

    for _ in 1:D_STEPS
        discr_loss = train_discriminator!(x, y, gen, discr, opt, device)
    end

    for _ in 1:G_STEPS
        gen_loss = train_generator!(x, y, gen, discr, opt, device)
    end

    return (discr_loss, gen_loss)
end

function train(data, gen, discr, opt_gen, opt_discr; hparams)
    """
    data : list of filenames containing `train` images
           `get_batch(data[i])` should load the files
    """
    # Define Optimizers
    opt_gen = ADAM(hparams.gen_lr, (0.5, 0.999))
    opt_disc = ADAM(hparam.discr_lr, (0.5, 0.999))

    # Partition into minibatches
    mb_idxs = partition(shuffle!(collect(1:length(data))), hparams.batch_size)
    train_batches = [data[i] for i in mb_idxs]

    for epoch in 1:hparams.epochs
        for i in 1:length(train_batches)
            x, y = get_batch(train_batches[i], hparams.img_size)

            x = x |> hparams.device
            y = y |> hparams.device

            (discr_loss, gen_loss) = train_step(x, y, gen, discr, opt_gen, opt_discr, hparams.device; hparams.D_STEPS, hparams.G_STEPS)

            # Do some logging
            # Do some saving

    return (gen, discr)
end
