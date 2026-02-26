using Distributed
addprocs(20)
@everywhere using OrdinaryDiffEq
@everywhere using HypergeometricFunctions, Optim, Statistics, Distributions, Plots
@everywhere using StatsBase, FastGaussQuadrature, LinearAlgebra
@everywhere using CSV,DataFrames
@everywhere using NLsolve
@everywhere using JSON,TaylorSeries,MultiFloats
@everywhere using SparseArrays

@everywhere function cus_hist(data::Vector)
    max1 = maximum(data)
    edge1 = collect(0:1:max1+1)
    h = fit(Histogram, data,edge1.-0.5)
    Weights = h.weights/length(data)
    return Weights
end

# Telegrah model(with time) FSP 
@everywhere function fsp1(params,N,τ)
    ρ,σon,σoff,d=min.(100,params)
    N=Int(N)
    # Define transition matrix for \bar{P}'s CME
    C1 = - spdiagm(0 => σon*ones(N)) - spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
    C2 =  spdiagm(0 => σoff*ones(N)) 
    C3 = spdiagm(0 => σon*ones(N)) 
    C4 = - spdiagm(0 => σoff*ones(N))  - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))- spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
    C = zeros(2*N,2*N)
    C[1:N,1:N] = C1
    C[1:N,N+1:2*N] = C2
    C[N+1:2*N,1:N] = C3
    C[N+1:2*N,N+1:2*N] = C4
    tspan = (0.0,τ)
    u_aux = zeros(2*N)
    u_aux[N+1] = 1.
    f(u,p,t)=C*u
    prob1 = ODEProblem(f, u_aux, tspan)
    # sol1 = solve(prob1, Tsit5(), saveat=0.01)
    sol1 = solve(prob1, Rodas5(), saveat=0.01)
    return sol1.u
end

@everywhere function calculate_realative_mse(est_params, true_params)
    return mean(abs.(est_params - true_params)./true_params)
end

@everywhere function objective_func(params, t, weight, sol_lis)
    n = length(weight)
    p_ge = sol_lis  
    a = zeros(length(p_ge) - n)
    weight = [weight; a]
    p_ge = max.(0.0, p_ge)
    p_ge = p_ge ./ sum(p_ge)
    eps = 1e-20
    p_ge = max.(p_ge, eps)
    return -sum(weight .* log.(p_ge))
end

    
#Telegrah model(with time) PGF
@everywhere function model_gf(p,t,z)
    p = min.(100,p)
    d = p[4]
    # d=1
    ρ = p[1]/d
    σon = p[2]/d
    σoff = p[3]/d
    Σ = σon+σoff+1
    T = d*t
    W = (z-1)*exp(-T)
    w = z-1
    f = σoff/(Σ-1)*exp(-ρ*W)*HypergeometricFunctions.pFq((σon,),(Σ,),ρ*W)
    g = σon/(Σ-1)*exp(-ρ*W)*HypergeometricFunctions.pFq((-σoff,),(2-Σ,),ρ*W)
    G0 = f*exp(-T*(σon+σoff))*HypergeometricFunctions.pFq((1-σoff,),(2-Σ,),ρ*w)+g*HypergeometricFunctions.pFq((1+σon,),(Σ,),ρ*w)
    G1 = -f*exp(-T*(σon+σoff))*HypergeometricFunctions.pFq((-σoff,),(2-Σ,),ρ*w)+σoff/σon*g*HypergeometricFunctions.pFq((σon,),(Σ,),ρ*w)
    return G0+G1
end
@everywhere mpgf(p,t) = (z->model_gf(p,t,z)).(zo)


# Empirical generating function obtained from histogram
@everywhere function hist_gf(hist_data,z)
    Nx = size(hist_data,1)
    z_vec = [z.^i for i = 0 : Nx-1]
    return sum(z_vec.*hist_data)
end
@everywhere epgf(his)=(z->hist_gf(his,z)).(zo)

@everywhere function sdist(hist_data,ps,t,xo,wo)
    mtgf = mpgf(ps,t)
    etgf = epgf(hist_data)
    return sum(wo.*(mtgf-etgf).^2)
end

@everywhere function obj_pgf(params,weights,time,xo,wo)
    total_err = sum([sdist(weights[l],params,time[l],xo,wo) for l in 1:length(time)])
    return total_err
end

@everywhere function obj_fsp(params,weights,time)
    N = 100
    sol1 = fsp1(params,N,7)
    sol_list =[]
    for ti in time 
        callback= Int(round(ti/0.01))+1
        push!(sol_list,sol1[callback][1:N]+sol1[callback][N+1:end])
    end
    total_err = sum([objective_func(params,time[l],weights[l],sol_list[l]) for l in 1:length(time)])
    return total_err
end

@everywhere function main_pgf(data, true_param, time, xo, wo; f_tol=1e-8, patience=60, max_iters=1000)
    weights = JSON.parse.(data)
    weights = [Float64.(aa) for aa in weights]
    ini_params = zeros(4)

    prev_f = Ref(Inf)
    stable_count = Ref(0)

    obj_func(params) = obj_pgf(exp.(params), weights, time, xo, wo)

    function my_callback(state)
        curr_f = state.value
        delta_f = abs(curr_f - prev_f[])./abs(prev_f[])
        println("📌 Callback: f = $curr_f | Δf = $delta_f")

        if delta_f < f_tol
            stable_count[] += 1
            println("⚠️ Δf < f_tol ($stable_count[]/$patience)")
            if stable_count[] ≥ patience
                println("✅ Early stopping triggered by callback.")
                return true 
            end
        elseif delta_f>1e-2
            stable_count[] = 0
        end

        prev_f[] = curr_f
        return false
    end

    elapsed_time = @elapsed begin
        result = optimize(
            obj_func,
            ini_params,
            NelderMead(),
            Optim.Options(
                iterations = max_iters,
                g_tol=1e-20,
                show_trace = true,
                callback = my_callback
            )
        )
    end

    p = exp.(Optim.minimizer(result))
    mse = calculate_realative_mse(p, true_param)
    return [mse, elapsed_time, p]
end

@everywhere function main_fsp(data, true_param, time; f_tol=1e-8, patience=60, max_iters=1000)
    weights = JSON.parse.(data)
    weights = [Float64.(aa) for aa in weights]
    ini_params = zeros(4)

    prev_f = Ref(Inf)
    stable_count = Ref(0)

    obj_func(params) = obj_fsp(exp.(params), weights, time)

    function my_callback(state)
        curr_f = state.value
        delta_f = abs(curr_f - prev_f[]) / (abs(prev_f[]) + 1e-8)
        println("📌 Callback: f = $curr_f | Δf = $delta_f")

        if delta_f < f_tol
            stable_count[] += 1
            println("⚠️ Δf < f_tol ($stable_count[]/$patience)")
            if stable_count[] ≥ patience
                println("✅ Early stopping triggered by callback.")
                return true 
            end
        elseif delta_f>1e-2
            stable_count[] = 0
        end

        prev_f[] = curr_f
        return false
    end
    
    elapsed_time = @elapsed begin
        result = optimize(
            obj_func,
            ini_params,
            NelderMead(),
            Optim.Options(
                iterations = max_iters,
                g_tol = 1e-20,
                show_trace = true,
                callback = my_callback
            )
        )
    end

    p = exp.(Optim.minimizer(result))
    mse = calculate_realative_mse(p, true_param)
    return [mse, elapsed_time, p]
end

x, w = gausslegendre(5)
min_z = 0.9; max_z = 1.
zo = (max_z-min_z)/2 .* x .+ (max_z+min_z)/2
wo = w * (max_z-min_z)/2
#cell number list
nc_list = [100,300,1000,3000,6000]  
#time points list
time_list = [collect(0.:6/(Int(12000/nc)):6) for nc in nc_list]

data_l1_10000 = CSV.read("l2-data.csv",DataFrame)
data_l4_10000 = CSV.read("l4-data.csv",DataFrame)
data_l12_10000 = CSV.read("l12-data.csv",DataFrame)
data_l40_10000 = CSV.read("l40-data.csv",DataFrame)
data_l120_10000 = CSV.read("l120-data.csv",DataFrame)
param10000 =Matrix(CSV.read("params.csv",DataFrame))

res_pgf = pmap(gj->main_pgf(data_l1_10000[2:end,gj],param10000[:,gj],time_list[5][2:end],zo,wo),1:50) 
res_fsp = pmap(gj->main_fsp(data_l1_10000[2:end,gj],param10000[:,gj],time_list[5][2:end]),1:50)
a0 = [median(hcat(res_pgf...)[1,:][(j-1)*10+1:j*10]) for j = 1:ng]
b0 = [median(hcat(res_fsp...)[1,:][(j-1)*10+1:j*10]) for j = 1:ng]

