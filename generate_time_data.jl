using Distributed
addprocs(20)
@everywhere using OrdinaryDiffEq
@everywhere using HypergeometricFunctions, Optim, Statistics, Distributions, Plots
@everywhere using StatsBase, FastGaussQuadrature, LinearAlgebra
@everywhere using CSV,DataFrames
@everywhere using NLsolve
@everywhere using OptimalTransport
@everywhere using JSON,TaylorSeries,MultiFloats
@everywhere using SparseArrays

@everywhere function cus_hist(data::Vector)
    max1 = maximum(data)
    edge1 = collect(0:1:max1+1)
    h = fit(Histogram, data,edge1.-0.5)
    Weights = h.weights/length(data)
    return Weights
end

#SSA initial at G=0, ntem=[P,G,G*]
@everywhere function propensity(params,Np)
    f_r = zeros(4)
    ρ,σon,σoff,d = params
    #ρ d σoff σon
    f_r[1]= Np[2]*ρ
    f_r[2]= Np[1]*d
    f_r[3]= Np[2]*σoff
    f_r[4]= Np[3]*σon
    return f_r
end

# Single realization SSA
@everywhere function SSA(params,gj)
    # Number of reactions
    M = 4
    # Number of reactants
    N = 3
    # System size
    omega = 1
    #ρ d σoff σon
    #G->G+P3
    S1=[1,1,0]-[0,1,0]
    #P->None
    S2=[0,0,0]-[1,0,0]
    #G->G*
    S3=[0,0,1]-[0,1,0]
    #G*->G
    S4=[0,1,0]-[0,0,1]

    # Define stoichiometry matrix
    S_mat = zeros(M,N)
    S_mat[1,1:N] = S1
    S_mat[2,1:N] = S2
    S_mat[3,1:N] = S3
    S_mat[4,1:N] = S4
    # Simulation duration
    tol_time = 6.2
    # Trajectory sampling period
    sp = 0.05
    # Define reactants trjatory vector
    n = zeros(N,Int(floor(tol_time/sp)));
    n_temp = [0,1,0];
    T = 0;
    f_r = zeros(M);

    while T < tol_time
        # Step 1: Calculate propensity
        f_r = propensity(params,n_temp)
        lambda = sum(f_r);
        # Step 2: Calculate tau and mu using random number genrators
        r1 = rand(2,1);
        tau = (1/lambda)*log(1/r1[1]);
        next_r=findfirst(x -> x>=r1[2]*lambda,cumsum(f_r));
        # Step 3: Updata the systemr
        # Update the time
        T += tau;
        # Update the trajectory vector
        if T <= tol_time
            for t in Int(ceil((T-tau)/sp)) : Int(floor(T/sp))
                n[1:N,t+1] = n_temp;
            end
            # Fire reaction next_r
            prod = S_mat[next_r,1:N];
            for i in 1:N
                n_temp[i] = n_temp[i] + prod[i];
            end
        else
            for t in Int(ceil((T-tau)/sp)) : Int(floor(tol_time/sp)-1)
                n[1:N,t+1] = n_temp;
            end
        end
    end
    #P G G*
    return n[1,:]
end


ng = 100; lowerbound = 1e-2
tpara1 = [9.64 2.763 1.66;
8.76 0.97 0.19;
6.26 2.86 1.00;
7.24 2.32 0.80;
8.54 2.60 2.69
]

nc_list = [100,300,1000,3000,6000]  
time_list = [[2,4],[1,2,3,4],[0.5,1,1.5,2,2.5,3,3.5,4]]
l1=[];l4=[];l12=[];l40=[];l120=[]
tpara =[]
for nc in nc_list
    ng = 5; lowerbound = 1e-2
    # tpara1 represent ρ,σon,σoff,d
    tpara =[]
    for i = 1:ng
        for j = 1:10
            pa = tpara1[i,:]
            q=pmap(gj->SSA(pa,gj),1:nc)
            qo = hcat(q...)
    
            t = collect(0.:6/(Int(12000/nc)):6)
            qoo = qo[Int.(round.(t./0.05) .+1),:]
            qj = [qoo[m,:] for m = 1:length(t)]
            # qj = [rand.(Binomial.(qoo[m,:],0.5)) for m = 1:length(t)]
        
            histo = [cus_hist(qj[m]) for m = 1:length(t)]
            if length(t) == 3
                push!(l1,histo)
            elseif length(t)==5
                push!(l4,histo)
            elseif length(t)==13
                push!(l12,histo)
            elseif length(t)==41
                push!(l40,histo)
            elseif length(t)==121
                push!(l120,histo)
            end
        
            push!(tpara,tpara1[i,:])
        end
    end
end

CSV.write("params.csv",DataFrame(tpara,:auto))
CSV.write("l1-data.csv",DataFrame(l1,:auto))
CSV.write("l4-data.csv",DataFrame(l4,:auto))
CSV.write("l12-data.csv",DataFrame(l12,:auto))
CSV.write("l40-data.csv",DataFrame(l40,:auto))
CSV.write("l120-data.csv",DataFrame(l120,:auto))
