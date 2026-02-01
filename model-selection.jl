
#function to compute custom histogram
function cus_hist(data::Vector)
    max1 = maximum(data)
    edge1 = collect(0:1:max1+1)
    h = fit(Histogram, data,edge1.-0.5)
    Weights = h.weights/length(data)
    return Weights
end

#function for model generating function
function model_gf(ps,z,t,sel)
    if sel == 1 #Telegrah model
        ρ,σon,σoff,d = min.(1000,ps)
        ρ = ρ/d;σon = σon/d;σoff = σoff/d
        Σ = σon+σoff+1
        T = d*t
        W = (z-1)*exp(-T)
        w = z-1
        f = σoff/(Σ-1)*exp(-ρ*W)*HypergeometricFunctions.pFq((σon,),(Σ,),ρ*W)
        g = σon/(Σ-1)*exp(-ρ*W)*HypergeometricFunctions.pFq((-σoff,),(2-Σ,),ρ*W)
        G0 = f*exp(-T*(σon+σoff))*HypergeometricFunctions.pFq((1-σoff,),(2-Σ,),ρ*w)+g*HypergeometricFunctions.pFq((1+σon,),(Σ,),ρ*w)
        G1 = -f*exp(-T*(σon+σoff))*HypergeometricFunctions.pFq((-σoff,),(2-Σ,),ρ*w)+σoff/σon*g*HypergeometricFunctions.pFq((σon,),(Σ,),ρ*w)
        return G0+G1
    elseif sel == 2 #Refractory model
        ρ,σb,σu,λ,d = min.(1000,ps)
        u = z-1
        x = ρ*u/d 
        h = ρ*u/d*exp(-d*t)
        k = λ+σu+σb
        δ = sqrt(Complex((λ-σb)^2-2*(λ+σb)*σu+σu^2))
        c0 = σb*σu*exp(-h)/(σb*σu+λ*σb+λ*σu)*pFq((λ/d-k/(2*d)+δ/(2*d),σu/d-k/(2*d)+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),h)*pFq((λ/d-k/(2*d)-δ/(2*d),σu/d-k/(2*d)-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),h) +
        σb*σu*d/((σb*σu+λ*σb+λ*σu)*δ)*(2*λ-k+δ)*(2*σu-k+δ)/((2*d-k+δ)*(2*d+2*δ))*h*exp(-h)*pFq((1+λ/d-k/(2*d)+δ/(2*d),1+σu/d-k/(2*d)+δ/2/d),(2-k/2/d+δ/2/d,2+δ/d),h)*pFq((λ/d-k/(2*d)-δ/(2*d),σu/d-k/(2*d)-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),h)-
        σb*σu*d/((σb*σu+λ*σb+λ*σu)*δ)*(2*λ-k-δ)*(2*σu-k-δ)/((2*d-k-δ)*(2*d-2*δ))*h*exp(-h)*pFq((λ/d-k/(2*d)+δ/(2*d),σu/d-k/(2*d)+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),h)*pFq((1+λ/d-k/(2*d)-δ/(2*d),1+σu/d-k/(2*d)-δ/2/d),(2-k/2/d-δ/2/d,2-δ/d),h)

        c1 = -σb*σu*(k+δ)/(2*δ*(σb*σu+λ*σb+λ*σu))*exp(-h)*h^(-δ/2/d+k/2/d)*pFq((λ/d-k/(2*d)-δ/(2*d),σu/d-k/(2*d)-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),h)*pFq((λ/d,σu/d),(1+k/2/d-δ/2/d,1+k/2/d+δ/2/d),h)+
        σb*σu*d/((σb*σu+λ*σb+λ*σu)*δ)*(2*λ-k-δ)*(2*σu-k-δ)/((2*d-k-δ)*(2*d-2*δ))*h^(-δ/2/d+1+k/2/d)*exp(-h)*pFq((1+λ/d-k/(2*d)-δ/(2*d),1+σu/d-k/(2*d)-δ/2/d),(2-k/2/d-δ/2/d,2-δ/d),h)*pFq((λ/d,σu/d),(1+k/2/d-δ/2/d,1+k/2/d+δ/2/d),h)-
        σb*σu*d/((σb*σu+λ*σb+λ*σu)*δ)*4*λ*σu/((2*d+k-δ)*(2*d+k+δ))*h^(-δ/2/d+1+k/2/d)*exp(-h)*pFq((λ/d-k/(2*d)-δ/(2*d),σu/d-k/(2*d)-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),h)*pFq((1+λ/d,1+σu/d),(2+k/2/d-δ/2/d,2+k/2/d+δ/2/d),h)

        c2 = σb*σu*d/((σb*σu+λ*σb+λ*σu)*δ)*4*λ*σu/((2*d+k-δ)*(2*d+k+δ))*h^(δ/2/d+1+k/2/d)*exp(-h)*pFq((λ/d-k/(2*d)+δ/(2*d),σu/d-k/(2*d)+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),h)*pFq((1+λ/d,1+σu/d),(2+k/2/d-δ/2/d,2+k/2/d+δ/2/d),h)+
        σb*σu*d*(k-δ)/((σb*σu+λ*σb+λ*σu)*2*δ)*h^(δ/2/d+k/2/d)*exp(-h)*pFq((λ/d-k/(2*d)+δ/(2*d),σu/d-k/(2*d)+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),h)*pFq((λ/d,σu/d),(1+k/2/d-δ/2/d,1+k/2/d+δ/2/d),h)-
        σb*σu*d/((σb*σu+λ*σb+λ*σu)*δ)*(2*λ-k+δ)*(2*σu-k+δ)/((2*d-k+δ)*(2*d+2*δ))*h^(δ/2/d+1+k/2/d)*exp(-h)*pFq((1+λ/d-k/(2*d)+δ/(2*d),1+σu/d-k/(2*d)+δ/2/d),(2-k/2/d+δ/2/d,2+δ/d),h)*pFq((λ/d,σu/d),(1+k/2/d-δ/2/d,1+k/2/d+δ/2/d),h)

        w0 = pFq((λ/d,σu/d),(1+k/2/d-δ/2/d,1+k/2/d+δ/2/d),x)
        w1 = x^(-k/2/d+δ/2/d)*pFq((λ/d-k/2/d+δ/2/d,σu/d-k/2/d+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),x)
        w2 = x^(-k/2/d-δ/2/d)*pFq((λ/d-k/2/d-δ/2/d,σu/d-k/2/d-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),x)

        w01 = 4*x*λ*σu/((2*d+k-δ)*(2*d+k+δ))*pFq((1+λ/d,1+σu/d),(2+k/2/d-δ/2/d,2+k/2/d+δ/2/d),x)
        w11 = (-k/2/d+δ/2/d)*x^(-k/2/d+δ/2/d)*pFq((λ/d-k/2/d+δ/2/d,σu/d-k/2/d+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),x)+
        ((2*λ-k+δ)*(2*σu-k+δ))/((2*d-k+δ)*(2*d+2*δ))*x^(-k/2/d+1+δ/2/d)*pFq((1+λ/d-k/2/d+δ/2/d,1+σu/d-k/2/d+δ/2/d),(2-k/2/d+δ/2/d,2+δ/d),x)
        w21 = (-k/2/d-δ/2/d)*x^(-k/2/d-δ/2/d)*pFq((λ/d-k/2/d-δ/2/d,σu/d-k/2/d-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),x)+
        ((2*λ-k-δ)*(2*σu-k-δ))/((2*d-k-δ)*(2*d-2*δ))*x^(-k/2/d+1-δ/2/d)*pFq((1+λ/d-k/2/d-δ/2/d,1+σu/d-k/2/d-δ/2/d),(2-k/2/d-δ/2/d,2-δ/d),x)

        w02 = 4*x^2*λ*σu/((2*d+k-δ)*(2*d+k+δ))*4*(d+λ)*(d+σu)/((4*d+k-δ)*(4*d+k+δ))*pFq((2+λ/d,2+σu/d),(3+k/2/d-δ/2/d,3+k/2/d+δ/2/d),x)
        w12 = (-k/2/d+δ/2/d)*x^(-k/2/d+δ/2/d)*(-k/2/d+δ/2/d-1)*pFq((λ/d-k/2/d+δ/2/d,σu/d-k/2/d+δ/2/d),(1-k/2/d+δ/2/d,1+δ/d),x)+
        ((2*λ-k+δ)*(2*σu-k+δ))/((2*d-k+δ)*(d+δ))*(-k/2/d+δ/2/d)*x^(-k/2/d+1+δ/2/d)*pFq((1+λ/d-k/2/d+δ/2/d,1+σu/d-k/2/d+δ/2/d),(2-k/2/d+δ/2/d,2+δ/d),x)+
        (λ/d-k/2/d+δ/2/d)*(σu/d-k/2/d+δ/2/d)/((1-k/2/d+δ/2/d)*(1+δ/d))*(2*d+2*λ-k+δ)*(2*d+2*σu-k+δ)/((4*d-k+δ)*(4*d+2*δ))*x^(-k/2/d+2+δ/2/d)*pFq((2+λ/d-k/2/d+δ/2/d,2+σu/d-k/2/d+δ/2/d),(3-k/2/d+δ/2/d,3+δ/d),x)
        w22 = (-k/2/d-δ/2/d)*x^(-k/2/d-δ/2/d)*(-k/2/d-δ/2/d-1)*
        pFq((λ/d-k/2/d-δ/2/d,σu/d-k/2/d-δ/2/d),(1-k/2/d-δ/2/d,1-δ/d),x)+
        ((2*λ-k-δ)*(2*σu-k-δ))/((2*d-k-δ)*(d-δ))*(-k/2/d-δ/2/d)*x^(-k/2/d+1-δ/2/d)*
        pFq((1+λ/d-k/2/d-δ/2/d,1+σu/d-k/2/d-δ/2/d),(2-k/2/d-δ/2/d,2-δ/d),x)+
        (2*λ-k-δ)*(2*σu-k-δ)/((2*d-k-δ)*(2*d-2*δ))*(2*d+2*λ-k-δ)*(2*d+2*σu-k-δ)/((4*d-k-δ)*(4*d-2*δ))*
        x^(-k/2/d+2-δ/2/d)*pFq((2+λ/d-k/2/d-δ/2/d,2+σu/d-k/2/d-δ/2/d),(3-k/2/d-δ/2/d,3-δ/d),x)

        G = (σb*σu+λ*σb+λ*σu)/(σb*σu)*(c0*w0+c1*w1+c2*w2)+(d+λ+σb+σu)*d/(σb*σu)*(c0*w01+c1*w11+c2*w21)+d^2/(σb*σu)*(c0*w02+c1*w12+c2*w22)
        return real(G)
    end
end

#function to compute generating function
function hist_gf(hist_data,z)
    Nx = size(hist_data,1)
    z_vec = [z.^i for i = 0 : Nx-1]
    return sum(z_vec.*hist_data)
end

#compute empirical generating function
epgf(his)=(z->hist_gf(his,z)).(zo)

#compute objective function on single time point
function sdist(hist_data,ps,t,xo,wo,sel)
    mtgf =(z->model_gf(ps,z,t,sel)).(xo)
    etgf = epgf(hist_data)
    return sum(wo.*(mtgf-etgf).^2)
end

#compute objective function
function obj_pgf(params,D1,time,xo,wo,sel)
    weights = [cus_hist(D1[l,:]) for l in 1:length(time)]
    total_err = sum([sdist(weights[l],params,time[l],xo,wo,sel) for l in 1:length(time)])
    return total_err
end

x, w = gausslegendre(5)
min_z = 0.9; max_z = 1.
zo = (max_z-min_z)/2 .* x .+ (max_z+min_z)/2
wo = w * (max_z-min_z)/2

function inf_err(D1,time,xo,wo,sel;f_tol=1e-8, patience=60)
    prev_f = Ref(Inf)
    stable_count = Ref(0)
    function my_callback(state)
        curr_f = state.value
        delta_f = abs(curr_f - prev_f[]) / abs(prev_f[])

        println("📌 Callback: f = $curr_f | Δf = $delta_f")

        if delta_f < f_tol
            stable_count[] += 1
            println("⚠️ Δf < f_tol ($stable_count[]/$patience)")
            if stable_count[] ≥ patience
                println("✅ Early stopping triggered by callback.")
                return true  # stop
            end
        elseif delta_f > 1e-2
            stable_count[] = 0
        end

        prev_f[] = curr_f
        return false
    end
    if sel == 1
        init_ps = zeros(4)
        results = optimize(ps->obj_pgf(exp.(ps),D1,time,xo,wo,sel),init_ps,Optim.Options(
            show_trace=true,g_tol=1e-20,iterations = 2000,
            )).minimizer
    elseif sel == 2
        init_ps = zeros(5)
        results = optimize(ps->obj_pgf(exp.(ps),D1,time,xo,wo,sel),init_ps,Optim.Options(
            show_trace=true,g_tol=1e-20,iterations = 2000,
            )).minimizer
    end 
    return exp.(results)
end


time = collect(0.:6/(Int(12000/1000)):6)[2:end]
data = Matrix(CSV.read("synthetic_data-l12.csv",DataFrame))
time = [6]
data = Matrix(CSV.read("synthetic_data_t6.csv",DataFrame))

chunk_size = round(Int,size(data,2)/10)
new_arrays_n=Vector[]
new_arrays_n = [data[:,j:min(j+chunk_size-1, end)] for j in 1:chunk_size:chunk_size*9]
new_arrays_n = push!(new_arrays_n, data[:,chunk_size*9+1:end])

err = zeros(10,2)
ps = Vector{Vector{Float64}}()
for sel = 1 : 2
    for i = 1 : 10
        temp = collect(1:10)
        sig = .!(temp.==i)
        rda = hcat(new_arrays_n[sig]...)
        push!(ps,vec(inf_err(rda,time,zo,wo,sel)))
        err[i,sel] = obj_pgf(ps[i+(sel-1)*10],rda,time,zo,wo,sel)
    end
end

function model_select(err)
    aerr = mean(err,dims=1)'
    best_aerr,ind = findmin(aerr)
    best_std = std(err[:,ind[1]])
    tr = best_aerr .+ [best_std*sqrt(1-(cor(err[:,i],err[:,ind[1]]))) for i = 1 :2]
    best_model = 0
    flg = vec(Float64.(aerr .< tr))
    if flg == zeros(2)
        best_model = ind[1]
    else
        best_model,~ =  findmin(vcat(collect(1:2)[flg[:,1] .== 1],ind[1]))
    end
    return best_model
end


function err_cross(u_counts,s_counts)
    chunk_size = round(Int,length(u_counts)/10)
    new_arrays_n=Vector[]
    new_arrays_n = [u_counts[j:min(j+chunk_size-1, end)] for j in 1:chunk_size:chunk_size*9]
    new_arrays_n = push!(new_arrays_n, u_counts[chunk_size*9+1:end])
    new_arrays_m=Vector[]
    new_arrays_m = [s_counts[j:min(j+chunk_size-1, end)] for j in 1:chunk_size:chunk_size*9]
    new_arrays_m = push!(new_arrays_m, s_counts[chunk_size*9+1:end])
    err = zeros(10,2)
    ps = Vector{Vector{Float64}}()
    for sel = 1 : 2
        for i = 1 : 10
            temp = collect(1:10)
            sig = .!(temp.==i)
            rdb = vec(vcat(new_arrays_m[sig]...))
            rda = vec(vcat(new_arrays_n[sig]...))
            push!(ps,vec(inf_err2(rda,rdb,sel)))
            err[i,sel] = int_dist2(ps[i+(sel-1)*10],cus_hist2(new_arrays_n[i],new_arrays_m[i]),1,X,W,sel)
        end
    end
    model_select(err)
    return model_select(err)
end

model_select(err)
err_cross(u_counts,s_counts)
