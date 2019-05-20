# weight initialization
function random_normal(shape...)
    # return map(Float32,zeros(shape...))
    return Flux.glorot_uniform(shape...)
end

ConvBlock(in_ch::Int,out_ch::Int) = 
    Chain(Conv((4,4), in_ch=>out_ch,pad = (1, 1), stride=(2,2);init=random_normal),
          BatchNormWrap(out_ch)...,
          x->leakyrelu.(x,0.2))

function add_conv_block(model,in_ch,out_ch)
    return Chain(model...,ConvBlock(in_ch,out_ch))
end

function Discriminator()
    model = Chain(Conv((4,4), 6=>64,pad = (1, 1), stride=(2,2);init=random_normal),BatchNormWrap(64)...,x->leakyrelu.(x,0.2),
                  ConvBlock(64,128),
                  ConvBlock(128,256),
                  ConvBlock(256,512),
                  ConvBlock(512,256),
                  ConvBlock(256,128),
                  ConvBlock(128,64),
                  Conv((4,4), 64=>1,pad = (1, 1), stride=(2,2);init=random_normal),
                  x->σ.(x))
    return model 
end