using Images
using ImageView
using CUDA
using CUDA.CUFFT
using Plots

function loadholo(path::String)
    out = Float32.(channelview(Gray.(load(path))))
end

function CuTransSqr!(Plane, datLen, wavLen, dx)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[y,x] = 1.0 - ((x-datLen/2)*wavLen/datLen/dx)^2 - ((y-datLen/2)*wavLen/datLen/dx)^2
    end
    return nothing
end

function CuTransFunc!(Plane, d_sqrPart, z0, wavLen, datLen)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[y,x] = exp(2im*pi*(z0)/wavLen*sqrt(d_sqrPart[y,x]))
    end
    return nothing
end

function CuConvTransFunc!(Plane, d_sqrPart, z0, wavLen, datLen,dx)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[y,x] = exp(2.0im*pi/wavLen*(z0 + ((x-datLen/2.0)^2*dx^2 + (y-datLen/2.0)^2*dx^2)/2.0/z0))/(1.0im*wavLen*z0)
    end
    return nothing
end

function CuUpdateImposed!(imposed, input, datLen)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        if input[y,x] < imposed[y,x]
            imposed[y,x] = input[y,x]
        end
    end
    return nothing
end

function CuPhaseRetrieval!(Plane, img1, img2, trans, transInv, iterations, datLen)
    compAmp1 = CuArray{ComplexF32}(undef,(datLen,datLen))
    compAmp2 = CuArray{ComplexF32}(undef,(datLen,datLen))
    phi1 = CUDA.ones(datLen,datLen)
    phi2 = CuArray{Float32}(undef,(datLen,datLen))

    sqrtImg1 = sqrt.(img1)
    sqrtImg2 = sqrt.(img2)

    compAmp1 = sqrtImg1.*1.0

    for itr in 1:iterations
        compAmp2 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(compAmp1)).*trans))
        # compAmp2 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(compAmp1)).*CUFFT.fftshift(CUFFT.fft(trans))))
        phi2 = angle.(compAmp2)
        # phi2 = atan.(imag.(compAmp2)./real.(compAmp2))

        compAmp2 = sqrtImg2.*exp.(1.0im.*phi2)

        compAmp1 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(compAmp2)).*transInv))
        # compAmp1 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(compAmp2)).*CUFFT.fftshift(CUFFT.fft(transInv))))
        # phi1 = atan.(imag.(compAmp1)./real.(compAmp1))
        phi1 = angle.(compAmp1)

        compAmp1 = sqrtImg1.*exp.(1.0im.*phi1)
    end

    Plane .= compAmp1
    return nothing
end

function CuBallLensField!(Plane, diam, noil, nlens, wavLen, datLen,dx)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if x <= datLen && y <= datLen
        if (x-datLen/2.0)^2+(y-datLen/2.0)^2 <= diam^2/4/dx/dx
            Plane[y,x] = exp(1.0im*(2.0*CUDA.pi/wavLen*noil*diam + 2.0*CUDA.pi/wavLen*(nlens-noil)*(diam/2.0 + sqrt( (diam/2)^2 - ((dx*(x-datLen/2.0))^2+(dx*(y-datLen/2.0))^2) ))))
            # Plane[y,x] = 0.0
            # Plane[y,x] = CUDA.exp(1.0im*2.0*CUDA.pi/wavLen*noil*diam)
        else
            Plane[y,x] = exp(1.0im*2.0*CUDA.pi/wavLen*noil*diam)
            # Plane[y,x] = 1.0
        end
    end

    return nothing
end

function CuHalfBallLensField!(Plane, diam, noil, nlens, wavLen, datLen,dx)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y

    if x <= datLen && y <= datLen
        if sqrt((x-datLen/2)^2+(y-datLen/2)^2) <= diam/2/dx
            Plane[y,x] = exp(1.0im*(2.0*CUDA.pi/wavLen*noil*diam + 2.0*CUDA.pi/wavLen*(nlens-noil)*sqrt( (diam/2)^2 - ((dx*(x-datLen/2.0))^2+(dx*(y-datLen/2.0))^2) )))
            # Plane[y,x] = 0.0
            # Plane[y,x] = CUDA.exp(1.0im*2.0*CUDA.pi/wavLen*noil*diam)
        else
            Plane[y,x] = exp(1.0im*2.0*CUDA.pi/wavLen*noil*diam)
            # Plane[y,x] = 1.0
        end
    end

    return nothing
end

function main()
    wavLen = 0.6328 # um
    datLen = 1024*2 # pixels
    dx = 10.0 # um
    z1 = 180000.0 # um
    z2 = 240000.0 # um
    dz = z2 - z1 # um
    diam = 500.0 # um
    noil = 1.51253
    nlens = 1.51509

    threads = (32,32)
    blocks = (cld(datLen,32),cld(datLen,32))

    sqr = CuArray{Float32}(undef,(datLen,datLen))
    transz1 = CuArray{ComplexF32}(undef,(datLen,datLen))
    transz2 = CuArray{ComplexF32}(undef,(datLen,datLen))
    transdz = CuArray{ComplexF32}(undef,(datLen,datLen))
    transInvdz = CuArray{ComplexF32}(undef,(datLen,datLen))
    blField = CuArray{ComplexF32}(undef,(datLen,datLen))
    prField = CuArray{ComplexF32}(undef,(datLen,datLen))
    @cuda threads = threads blocks = blocks CuHalfBallLensField!(blField, diam,noil,nlens,wavLen,datLen,dx)
    @cuda threads = threads blocks = blocks CuTransSqr!(sqr,datLen,wavLen,dx)
    @cuda threads = threads blocks = blocks CuTransFunc!(transz1,sqr,z1-diam,wavLen,datLen)
    @cuda threads = threads blocks = blocks CuTransFunc!(transz2,sqr,z2-diam,wavLen,datLen)
    @cuda threads = threads blocks = blocks CuTransFunc!(transdz,sqr,dz,wavLen,datLen)
    @cuda threads = threads blocks = blocks CuTransFunc!(transInvdz,sqr,-dz,wavLen,datLen)

    # @cuda threads = threads blocks = blocks CuConvTransFunc!(transz1,sqr,z1-diam,wavLen,datLen,dx)
    # @cuda threads = threads blocks = blocks CuConvTransFunc!(transz2,sqr,z2-diam,wavLen,datLen,dx)
    # @cuda threads = threads blocks = blocks CuConvTransFunc!(transdz,sqr,dz,wavLen,datLen,dx)
    # @cuda threads = threads blocks = blocks CuConvTransFunc!(transInvdz,sqr,-dz,wavLen,datLen,dx)

    d_holo1 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*transz1))
    d_holo2 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*transz2))

    # d_holo1 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*CUFFT.fftshift(CUFFT.fft(transz1))))
    # d_holo2 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*CUFFT.fftshift(CUFFT.fft(transz2))))

    d_img1 = abs.(d_holo1.*conj(d_holo1))
    d_img2 = abs.(d_holo2.*conj(d_holo2))

    CuPhaseRetrieval!(prField,d_img1,d_img2,transdz,transInvdz,11,datLen)

    transInvz1 = CuArray{ComplexF32}(undef,(datLen,datLen))
    @cuda threads = threads blocks = blocks CuTransFunc!(transInvz1,sqr,-z1+diam,wavLen,datLen)
    # @cuda threads = threads blocks = blocks CuConvTransFunc!(transInvz1,sqr,-z1+diam,wavLen,datLen,dx)
    reconstPlane = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(prField)).*transInvz1))
    # reconstPlane = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(prField)).*CUFFT.fftshift(CUFFT.fft(transInvz1))))
    # display(reconstPlane)
    reconstPhase = angle.(reconstPlane)
    # reconstPhase = CUDA.atan.(imag.(reconstPlane)./real.(reconstPlane))
    hostPhasePlane = Array(reconstPhase)

    out = Array(abs.(reconstPlane.*conj.(reconstPlane)))
    out = out./2.0
    out[1,1] = 0.99
    imshow(out[512:1535,512:1535])
    x = collect(1:1024)
    y = hostPhasePlane[512,512:1535] .- 2.0*pi/wavLen*noil*diam
    plot(x,y)
    # return hostPhasePlane
end

# main()

wavLen = 0.6328 # um
datLen = 1024*2 # pixels
dx = 10.0 # um
z1 = 180000.0 # um
z2 = 240000.0 # um
dz = z2 - z1 # um
diam = 500.0 # um
noil = 1.51253
nlens = 1.51509

threads = (32,32)
blocks = (cld(datLen,32),cld(datLen,32))

sqr = CuArray{Float32}(undef,(datLen,datLen))
transz1 = CuArray{ComplexF32}(undef,(datLen,datLen))
transz2 = CuArray{ComplexF32}(undef,(datLen,datLen))
transdz = CuArray{ComplexF32}(undef,(datLen,datLen))
transInvdz = CuArray{ComplexF32}(undef,(datLen,datLen))
blField = CuArray{ComplexF32}(undef,(datLen,datLen))
prField = CuArray{ComplexF32}(undef,(datLen,datLen))
@cuda threads = threads blocks = blocks CuHalfBallLensField!(blField, diam,noil,nlens,wavLen,datLen,dx)
@cuda threads = threads blocks = blocks CuTransSqr!(sqr,datLen,wavLen,dx)
@cuda threads = threads blocks = blocks CuTransFunc!(transz1,sqr,z1-diam,wavLen,datLen)
@cuda threads = threads blocks = blocks CuTransFunc!(transz2,sqr,z2-diam,wavLen,datLen)
@cuda threads = threads blocks = blocks CuTransFunc!(transdz,sqr,dz,wavLen,datLen)
@cuda threads = threads blocks = blocks CuTransFunc!(transInvdz,sqr,-dz,wavLen,datLen)

# @cuda threads = threads blocks = blocks CuConvTransFunc!(transz1,sqr,z1-diam,wavLen,datLen,dx)
# @cuda threads = threads blocks = blocks CuConvTransFunc!(transz2,sqr,z2-diam,wavLen,datLen,dx)
# @cuda threads = threads blocks = blocks CuConvTransFunc!(transdz,sqr,dz,wavLen,datLen,dx)
# @cuda threads = threads blocks = blocks CuConvTransFunc!(transInvdz,sqr,-dz,wavLen,datLen,dx)

d_holo1 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*transz1))
d_holo2 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*transz2))

# d_holo1 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*CUFFT.fftshift(CUFFT.fft(transz1))))
# d_holo2 = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(blField)).*CUFFT.fftshift(CUFFT.fft(transz2))))

d_img1 = abs.(d_holo1.*conj(d_holo1))
d_img2 = abs.(d_holo2.*conj(d_holo2))

CuPhaseRetrieval!(prField,d_img1,d_img2,transdz,transInvdz,11,datLen)

transInvz1 = CuArray{ComplexF32}(undef,(datLen,datLen))
@cuda threads = threads blocks = blocks CuTransFunc!(transInvz1,sqr,-z1+diam,wavLen,datLen)
# @cuda threads = threads blocks = blocks CuConvTransFunc!(transInvz1,sqr,-z1+diam,wavLen,datLen,dx)
reconstPlane = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(prField)).*transInvz1))
# reconstPlane = CUFFT.ifft(CUFFT.fftshift(CUFFT.fftshift(CUFFT.fft(prField)).*CUFFT.fftshift(CUFFT.fft(transInvz1))))
# display(reconstPlane)
reconstPhase = angle.(reconstPlane)
# reconstPhase = CUDA.atan.(imag.(reconstPlane)./real.(reconstPlane))
hostPhasePlane = Array(reconstPhase)

out = Array(abs.(reconstPlane.*conj.(reconstPlane)))
out = out./2.0
out[1,1] = 0.99
imshow(out[512:1535,512:1535])
x = collect(1:1024)
y = hostPhasePlane[512,512:1535] .- 2.0*pi/wavLen*noil*diam
plot(x,y)
# return hostPhasePlane

