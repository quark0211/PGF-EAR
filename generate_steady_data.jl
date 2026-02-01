using Optim, LinearAlgebra, Distributions, SparseArrays, Statistics,StatsBase,Random
using HypergeometricFunctions
using CSV,DataFrames

nc_list = [100,300,1000,3000,10000]
ng = 5

tpara1 = [9.64 2.763 1.66;
8.76 0.97 0.19;
6.26 2.86 1.00;
7.24 2.32 0.80;
8.54 2.60 2.69
]
tpara1 = repeat(tpara1, inner=(10,1))

#Generate steady data 
for nc in nc_list
    rd = zeros(nc,ng*10)
    for i = 1 : nc         
        rtemp = rand.(Beta.(tpara1[:,2],tpara1[:,3]))        
        rd[i,:] = rand.(Poisson.(rtemp.*tpara1[:,1]))
    end
    CSV.write("steady-state-data/steady-state-data$(nc).csv",DataFrame(rd,:auto))
    CSV.write("steady-state-data/steady-state-param$(nc).csv",DataFrame(tpara1,:auto))
end

#Generate steady data with 50% contamination
for nc in nc_list
    rd = zeros(nc,ng*10)
    for i = 1 : nc         
        rtemp = rand.(Beta.(tpara1[:,2],tpara1[:,3]))        
        rd[i,:] = rand.(Binomial.(rand.(Poisson.(rtemp.*tpara1[:,1])),0.5))
    end
    CSV.write("steady-state-data/steady-state-con50-data$(nc).csv",DataFrame(rd,:auto))
    CSV.write("steady-state-data/steady-state-con50-param$(nc).csv",DataFrame(tpara1,:auto))
end