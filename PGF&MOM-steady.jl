using Distributed
addprocs(14)
@everywhere using Distributions,TaylorSeries,HypergeometricFunctions,MultiFloats
@everywhere using StatsBase,LinearAlgebra,DataFrames,CSV
@everywhere using Optim

@everywhere function cus_hist(data::Vector)
    max1 = maximum(data)
    edge1 = collect(0:1:max1+1)
    h = fit(Histogram, data,edge1.-0.5)
    Weights = h.weights/length(data)
    return Weights
end

@everywhere function model_gf(ps,z)
    ρ,σon,σoff = min.(100.0,ps)
    return HypergeometricFunctions.pFq((σon,),(σon+σoff,),ρ*(z-1))
end

# Empirical generating function obtained from histogram
@everywhere function hist_gf(hist_data,z)
    Nx = size(hist_data,1)
    z_vec = [z.^i for i = 0 : Nx-1]
    return sum(z_vec.*hist_data)
end

@everywhere mpgf(p) = (z->model_gf(p,z)).(zo)
@everywhere epgf(his) = (z->hist_gf(his,z)).(zo)


@everywhere function sdist(hist_data,ps,z,wo)
    mgf =mpgf(ps)
    egf = epgf(hist_data)
    return sum(wo.*(mgf-egf).^2)
end

@everywhere function calculate_realative_mse(est_params, true_params)
    return mean(abs.(est_params - true_params)./true_params)
end

@everywhere function main(data,true_param,zo,wo)
    hist_data = cus_hist(data)
    ini_params=zeros(3)
    elapsed_time = @elapsed begin
        results_PGF = optimize(ps->sdist(hist_data,exp.(ps),zo,wo),ini_params,NelderMead(),Optim.Options(show_trace=false,g_tol=1e-20,iterations = 2000)).minimizer
    end
    p = exp.(results_PGF)
    mse=calculate_realative_mse(p,true_param)
    return [mse,elapsed_time,p]
end

@everywhere function mom_cal(X)
    # === Observed moments ===
    N = length(X)
    μ1_obs = mean(X)
    xc = X .- μ1_obs
    μ2_obs = mean(xc .^ 2)
    μ3_obs = mean(xc .^ 3)
    μ4_obs = mean(xc .^ 4)
    μ6_obs = mean(xc .^ 6)

    μ_obs = [μ1_obs, μ2_obs, μ3_obs]
    # === Variance estimation ===
    σ2 = zeros(3)
    σ2[1] = (1 / N) * μ2_obs^2
    σ2[2] = (1 / N) * (μ4_obs - ((N - 3) / (N - 1)) * μ2_obs^2)
    σ2[3] = (1 / N) * (μ6_obs - μ3_obs^2)
    return μ_obs,σ2
end

@everywhere function MOM(μ_obs,σ2,ps)
    ρ, σon, σoff = ps
    Σ = σon + σoff
    r1 = ρ * σon / Σ
    r2 = ρ^2 * σon * (1 + σon) / (Σ * (Σ + 1))
    r3 = ρ^3 * σon * (1 + σon) * (2 + σon) / (Σ * (Σ + 1) * (Σ + 2))
    μ1_model = r1
    μ2_model = r2 + r1
    μ3_model = r3 + 3r2 + r1

    μ2_center_model = μ2_model - μ1_model^2
    μ3_center_model = μ3_model - 3*μ2_model*μ1_model + 2*μ1_model^3

    μ_model = [μ1_model, μ2_center_model, μ3_center_model]
    loss = sum(((μ_obs[k] - μ_model[k])^2) / (σ2[k]) for k in 1:3)
    return loss
end

@everywhere function main_mom(data,true_param,zo,wo)
    μ_obs,σ2 = mom_cal(data)
    ini_params=zeros(3)
    elapsed_time = @elapsed begin
        results_PGF = optimize(ps->MOM(μ_obs,σ2,exp.(ps)),ini_params,NelderMead(),Optim.Options(show_trace=false,g_tol=1e-20,iterations = 2000)).minimizer
    end
    p = exp.(results_PGF)
    mse=calculate_realative_mse(p,true_param)
    return [mse,elapsed_time,p]
end


@everywhere using FastGaussQuadrature
x, w = gausslegendre(5)
min_z = 0.; max_z = 1.
zo = (max_z-min_z)/2 .* x .+ (max_z+min_z)/2
wo = w * (max_z-min_z)/2

ng = 5

# Load data and parameters
data_100 = Matrix(CSV.read("steady-state-data/steady-state-data100.csv",DataFrame))
param100 =Matrix(CSV.read("steady-state-data/steady-state-param100.csv",DataFrame))

#Run for PGF method
@time res100 = pmap(ng->main(data_100[:,ng],param100[ng,:],zo,wo),1:ng*10)
a1 = [median(hcat(res100...)[1,:][(i-1)*10+1:i*10]) for i = 1:ng]
mean(a1)

#Run for MOM method
res100 = pmap(ng->main_mom(data_100[:,ng],param100[ng,:],zo,wo),1:ng*10)
a1 = [median(hcat(res100...)[1,:][(i-1)*10+1:i*10]) for i = 1:ng]
mean(a1)