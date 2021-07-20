function random_crop(imgA,imgB,scale=256)
    W,H = size(imgA) # We have W = H
    diff = W - scale
    i = rand(1:diff)
    
    return imgA[i:i+scale-1,i:i+scale-1],imgB[i:i+scale-1,i:i+scale-1]
end

function random_jitter(img; RESIZE_SCALE = 286)
    A,B = img[:,1:256],img[:,257:end]
    
    A = imresize(A,(RESIZE_SCALE,RESIZE_SCALE))
    B = imresize(B,(RESIZE_SCALE,RESIZE_SCALE))
    
    A,B = random_crop(A,B)
    return cat(A,B,dims=2)
end

function load_image(filename, jitter=false)
    img = load(filename)
    if jitter
        img = random_jitter(img)
    end

    img = Float32.(channelview(img))
end

function load_dataset(path, imsize=256)
    imgs = []
    for r in readdir(path)
        img_path = string(path,r)
        push!(imgs,img_path)
    end
    imgs
end

function get_batch(files,imsize)
   """
   files : array of image names in a path
   """
   imgsA = []
   imgsB = []
   for file in files
        push!(imgsA,load_image(file)[:,:,1:256])
        push!(imgsB,load_image(file)[:,:,257:end])
   end
   return reshape(cat(imgsA...,dims=4),imsize,imsize,3,length(imgsA)),reshape(cat(imgsB...,dims=4),imsize,imsize,3,length(imgsB))
end

function make_minibatch(X, idxs,size=256)
    """
    size : Image dimension
    """
    X_batch = Array{Float32}(undef, size, size, 3, length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:, i] = Float32.(X[:,:,:,idxs[i]])
    end
    return X_batch
end

function norm(x)
    (2.0f0 .* x) .- 1.0f0
end

function denorm(x)
   (x .+ 1.0f0) ./ 2.0f0
end

function logitbinarycrossentropy(logŷ, y)
    mean((1 .- y).*logŷ .- logσ.(logŷ))
end

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
