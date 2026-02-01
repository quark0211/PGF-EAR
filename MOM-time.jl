using Distributed
addprocs(20)
@everywhere using OrdinaryDiffEq
@everywhere using HypergeometricFunctions, Optim, Statistics, Distributions, Plots
@everywhere using StatsBase, FastGaussQuadrature, LinearAlgebra
@everywhere using CSV,DataFrames
@everywhere using NLsolve
@everywhere using JSON,TaylorSeries,MultiFloats
@everywhere using SparseArrays


@everywhere function moemnt_ode(dy,y,p,t)
    ρ,σon,σoff,d = p 
    ρ = ρ/d;σon = σon/d;σoff = σoff/d
    np = y[1]
    ng = y[2]
    np2 = y[3]
    npng = y[4]
    dy[1] = ρ*ng - np 
    dy[2] = -σoff*ng + σon*(1-ng)
    dy[3] = 2*ρ*npng - 2*np2 + ρ*ng + np
    dy[4] = ρ*ng + σon*np - (σon+σoff+1)*npng 
end 


@everywhere function MOM(ps,μ_model,mom)
    μ_o = mom[1]; σ2 = mom[2]
    loss = sum([((μ_o[k]-μ_model[k])^2)/σ2[k] for k in 1:2])
    return loss
end

@everywhere function cal_mom(weight,N)
    x = collect(0:1:length(weight)-1)
    μ1 = sum(x.*weight)
    xc = x .- μ1 
    μ2 = sum(xc.^2 .* weight)
    μ4 = sum(xc.^4 .* weight)
    μ = [μ1,μ2]
    σ2 = zeros(2)
    σ2[1] = (1/N)* μ2^2
    σ2[2] = (1/N)*(μ4-(N-3)/(N-1)*μ2^2)
    return [μ,σ2]
end

@everywhere function obj_MOM(params,weights,time,N)
    y0 = [0,1,0,0]
    tspan = (0.,7)
    prob = ODEProblem(moemnt_ode,y0,tspan,params)
    sol = solve(prob,Rodas5(),saveat=0.01)
    sol_list =[]
    for ti in time 
        callback= Int(round(ti/0.01))+1
        push!(sol_list,[sol[callback][1],sol[callback][3]-sol[callback][1]^2])
    end
    moment = [cal_mom(weights[l],N) for l in 1:length(time)]
    total_err = sum([MOM(params,sol_list[l],moment[l]) for l in 1:length(time)])
    return total_err
end


@everywhere function main_mom(data, true_param, time, N; f_tol=1e-8, patience=60, max_iters=1000)
    weights = JSON.parse.(data)
    weights = [Float64.(aa) for aa in weights]
    # ini_params = zeros(4)
    ini_params = log.([1,1.1,1,1])

    prev_f = Ref(Inf)
    stable_count = Ref(0)
    obj_func(params) = obj_MOM(exp.(params), weights, time, N)
    function my_callback(state)
        curr_f = state.value
        delta_f = abs(curr_f - prev_f[])./abs(prev_f[])
        println("📌 Callback: f = $curr_f | Δf = $delta_f")

        if delta_f < f_tol
            stable_count[] += 1
            println("⚠️ Δf < f_tol ($stable_count[]/$patience)")
            if stable_count[] ≥ patience
                println("✅ Early stopping triggered by callback.")
                return true  # 提前终止
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
                # callback = my_callback
            )
        )
    end

    p = exp.(Optim.minimizer(result))
    mse = calculate_realative_mse(p, true_param)
    return [mse, elapsed_time, p]
end


data_l1_10000 = CSV.read("l2-data.csv",DataFrame)
data_l4_10000 = CSV.read("l4-data.csv",DataFrame)
data_l12_10000 = CSV.read("l12-data.csv",DataFrame)
data_l40_10000 = CSV.read("l40-data.csv",DataFrame)
data_l120_10000 = CSV.read("l120-data.csv",DataFrame)
param10000 =Matrix(CSV.read("params.csv",DataFrame))
nc_list = [100,300,1000,3000,6000]  
time_list = [collect(0.:6/(Int(12000/nc)):6) for nc in nc_list]

res_mom = pmap(gj->main_mom(data_l1_10000[2:end,gj],param10000[:,gj],time_list[5][2:end],12000),1:50)
a0 = [median(hcat(res_mom...)[1,:][(j-1)*10+1:j*10]) for j = 1:ng]
