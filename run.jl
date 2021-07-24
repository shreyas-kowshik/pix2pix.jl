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
using Printf
using Random
using CUDA

@with_kw struct HyperParams
    batch_size::Int = 128
    epochs::Int = 20
    img_size::Int = 256
    discr_lr::Float64 = 0.0002
    gen_lr::Float64 = 0.0002
    D_STEPS::Int = 1
    G_STEPS::Int = 1
    device = cpu
end

hparams = HyperParams()

dataset_path = "facades/train/"
num_examples = 10

# Data Loading
data = load_dataset(dataset_path, hparams.img_size)[1:num_examples] # data : list of filenames
print("Loaded Data!")


