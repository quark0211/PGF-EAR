

using Distributed
addprocs(19)
@everywhere using Distributions,TaylorSeries,HypergeometricFunctions,MultiFloats
@everywhere using StatsBase,LinearAlgebra,DataFrames,CSV
@everywhere using Optim




# Generate sample data from true distribution
@everywhere function fit_histogram(data)
    data2=data
    edge = collect(minimum(data2):1:maximum(data2)+1).-0.5
    h = fit(Histogram, data2, edge)
    weight = h.weights
    pop!(edge)
    node = edge.+0.5
    weight = weight./sum(weight)
    return node,weight
end


@everywhere function fit_init_params(data)
    initial_params=zeros(3)
    Ex = mean(data)
    Ex2 = mean(data.^2)
    Ex3 = mean(data.^3)
    initial_params[2] = 2*Ex*(-Ex^2+Ex2^2+Ex*(Ex2-Ex3))/(3*Ex^3+Ex2*(3*Ex2-Ex3)+2*Ex^2*(1-2*Ex2+Ex3)+Ex*(-Ex2*(5+Ex2)+Ex3))
    initial_params[3] = 2*(Ex+Ex^2-Ex2)*(Ex*(2+Ex-Ex2)-3*Ex2+Ex3)*(Ex^2-Ex2^2+Ex*(-Ex2+Ex3))/((Ex^3-Ex^2*Ex2+2*Ex2^2-Ex*(Ex2+Ex3))*(3*Ex^3+Ex2*(3*Ex2-Ex3)+2*Ex^2*(1-2*Ex2+Ex3)+Ex*(-Ex2*(5+Ex2)+Ex3)))
    initial_params[1] = (-3*Ex^3+Ex*Ex2*(5+Ex2)-Ex*Ex3+Ex2*(-3*Ex2+Ex3)+Ex^2*(4*Ex2-2*(1+Ex3)))/(Ex^3-Ex^2*Ex2+2*Ex2^2-Ex*(Ex2+Ex3))
    return initial_params
end

@everywhere function estimate_init_params(node,weight)
    initial_params=zeros(3)
    Ex = sum(node.*weight)
    p=node.^2
    Ex2 = sum(p.*weight)
    t=node.^3
    Ex3 = sum(t.*weight)
    initial_params[1] = 2*Ex*(-Ex^2+Ex2^2+Ex*(Ex2-Ex3))/(3*Ex^3+Ex2*(3*Ex2-Ex3)+2*Ex^2*(1-2*Ex2+Ex3)+Ex*(-Ex2*(5+Ex2)+Ex3))
    initial_params[2] = 2*(Ex+Ex^2-Ex2)*(Ex*(2+Ex-Ex2)-3*Ex2+Ex3)*(Ex^2-Ex2^2+Ex*(-Ex2+Ex3))/((Ex^3-Ex^2*Ex2+2*Ex2^2-Ex*(Ex2+Ex3))*(3*Ex^3+Ex2*(3*Ex2-Ex3)+2*Ex^2*(1-2*Ex2+Ex3)+Ex*(-Ex2*(5+Ex2)+Ex3)))
    initial_params[3] = (-3*Ex^3+Ex*Ex2*(5+Ex2)-Ex*Ex3+Ex2*(-3*Ex2+Ex3)+Ex^2*(4*Ex2-2*(1+Ex3)))/(Ex^3-Ex^2*Ex2+2*Ex2^2-Ex*(Ex2+Ex3))
    return initial_params
end


@everywhere function fsp(params,N)
    ρ,σon,σoff = params
    N=Int(N)
    D=diagm(0 => fill(-σon,N))-diagm(0 => collect(0:N-1))+diagm(1=>collect(1:N-1))
    C=diagm(0=>fill(σoff,N))
    B=diagm(0=>fill(σon,N))
    A=diagm(0=>fill(-σoff-ρ,N))-diagm(0 => collect(0:N-1))+diagm(-1=>fill(ρ,N-1))+diagm(1=>collect(1:N-1))
    AB=hcat(A,B)
    CD=hcat(C,D)
    matrix=vcat(AB,CD)
    matrix[end,:].=1.0
    u0=[zeros(N*2-1);1]
    p0=matrix\u0
    prob=p0[1:N]+p0[N+1:end]
    return prob
end


@everywhere function calculate_realative_mse(est_params, true_params)
    return mean(abs.(est_params - true_params)./true_params)
end

@everywhere function objective_func(params,node,weight)
    n = length(node)
    p_ge = fsp(params,101)
    a=zeros(length(p_ge)-n)
    weight = [weight;a]
    node=[i for i=0:1:length(p_ge)-1]
    p_ge=max.(1e-15,p_ge)
    p_ge=p_ge./sum(p_ge)
    return -sum(weight .* log.(p_ge))
end

@everywhere function main_fsp(data,true_param)
    node,weight = fit_histogram(data)
    # ini_params=fit_init_params(data)
    ini_params=zeros(3)
    elapsed_time = @elapsed begin
        p = optimize(params->objective_func(exp.(params),node,weight),ini_params,NelderMead(),Optim.Options(show_trace=false,iterations = 2000)).minimizer
    end
    p = exp.(p)
    mse=calculate_realative_mse(p,true_param)
    return [mse,elapsed_time,p]
end


ng=5
# Load data and parameters
data_100 = Matrix(CSV.read("steady-state-data/steady-state-data100.csv",DataFrame))
param100 =Matrix(CSV.read("steady-state-data/steady-state-param100.csv",DataFrame))

@time res100 = pmap(ng->main_fsp(data_100[:,ng],param100[ng,:]),1:ng*10)
a1 = [median(hcat(res100...)[1,:][(i-1)*10+1:i*10]) for i = 1:ng]
mean(a1)




