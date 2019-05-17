ConvBlock(in_ch::Int,out_ch::Int) = 
    Chain(Conv((4,4), in_ch=>out_ch,pad = (1, 1), stride=(2,2)),
          BatchNormWrap(out_ch)...,
          x->leakyrelu.(x))

function add_conv_block(model,in_ch,out_ch)
    return Chain(model...,ConvBlock(in_ch,out_ch))
end

function Discriminator()
    model = Chain(Conv((4,4), 6=>64,pad = (1, 1), stride=(2,2)),x->leakyrelu.(x),
                  ConvBlock(64,128),
                  ConvBlock(128,256),
                  ConvBlock(256,512),
                  ConvBlock(512,512),
                  ConvBlock(512,512),
                  ConvBlock(512,512),
                  Conv((4,4), 512=>1,pad = (1, 1), stride=(2,2)),
                  x->σ.(x))
    return model 
end