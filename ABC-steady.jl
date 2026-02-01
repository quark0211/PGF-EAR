using Distributed
addprocs(19)

@everywhere using ApproxBayes
@everywhere using Distributions,HypergeometricFunctions
@everywhere using StatsBase,LinearAlgebra,DataFrames,CSV
@everywhere using Optim

#
@everywhere function Teledist(params,constant,targetdata)
    rtemp = rand(Beta(params[2],params[3]),nc)
    rd = [rand(Poisson(rtemp[i]*params[1])) for i = 1:nc]
    ApproxBayes.ksdist(rd,targetdata), 1
end

@everywhere nc = 1000

@everywhere setup = ABCRejection(Teledist,3,0.1,Prior([Gamma(2,2),Gamma(2,2),Gamma(2,2)]);maxiterations = 10^8)

@everywhere function calculate_realative_mse(est_params, true_params)
    return mean(abs.(est_params - true_params)./true_params)
end


data_1000 = Matrix(CSV.read("steady-state-data/steady-state-data1000.csv",DataFrame))
param1000 =Matrix(CSV.read("steady-state-data/steady-state-param1000.csv",DataFrame))

@everywhere function main_abc(data,true_param)
    elapsed_time = @elapsed begin
        rejection = runabc(setup,data)
        p = vec(median(rejection.parameters,dims=1))
    end
    mse=calculate_realative_mse(p,true_param)
    return [mse,elapsed_time,p]
end
ng = 5
@time res1000 = pmap(ng->main_abc(data_1000[:,ng],param1000[ng,:]),1:ng*10)
a1 = [median(hcat(res1000...)[1,:][(i-1)*10+1:i*10]) for i = 1:ng]
mean(a1)

