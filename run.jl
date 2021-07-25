using Pkg
Pkg.activate(".")
using pix2pix
using Base.Iterators: partition
using Flux
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Parameters
using Random
using CUDA
using Distributions

@with_kw struct HyperParams
    batch_size::Int = 2
    epochs::Int = 100
    img_size::Int = 256
    discr_lr::Float64 = 0.0002f0
    gen_lr::Float64 = 0.0002f0
    verbose_freq::Int = 1
    D_STEPS::Int = 1
    G_STEPS::Int = 1
    device = cpu
end

# Define Networks #
# weight initialization
function random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

ConvBlock(in_ch::Int,out_ch::Int,k=4,s=2,p=1) = 
    Chain(Conv((3,3), in_ch=>out_ch,pad = (p, p), stride=(1,1);init=random_normal),
          BatchNorm(out_ch),
          x->leakyrelu.(x,0.2f0))

function Discriminator()
    model = Chain(Conv((4,4), 6=>64,pad = (1, 1), stride=(2,2);init=random_normal), BatchNorm(64), x->leakyrelu.(x,0.2f0),
                  ConvBlock(64,128),
                  ConvBlock(128,256),
                  ConvBlock(256,512,4,1,1),
                  Conv((4,4), 512=>1,pad = (1, 1), stride=(1,1);init=random_normal))
    return model 
end


hparams = HyperParams()

dataset_path = "facades/train/"
num_examples = 10

# Data Loading
data = load_dataset(dataset_path, hparams.img_size)[1:num_examples] # data : list of filenames
println("Loaded Data!")

gen = UNet() |> hparams.device
discr = Discriminator() |> hparams.device

(gen, discr) = train(data, gen, discr; hparams)
