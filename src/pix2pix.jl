module pix2pix

using CUDAnative
using Images,CuArrays,Flux
using Random
using Statistics
using JLD
using Plots

include("utils.jl")

export

load_dataset, load_image, get_batch, make_minibatch,
norm,

train

end # module
