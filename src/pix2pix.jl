module pix2pix

using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Parameters
using Printf
using Random
using CUDA

# using Images
# using Flux
# using CUDA
# using Random
# using Statistics
# using JLD
# using Plots
# using Parameters

include("utils.jl")
inculde("train.jl")
include("generator.jl")
include("discriminator.jl")

export

load_dataset, load_image, get_batch, make_minibatch,
norm,

train

end # module
