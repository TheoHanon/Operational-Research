using Distributions
using StatsBase


function read_file(path::String)
    """
    Reads a file and returns its contents as a Matrix
    """

    data = open(path, "r") do file
        [parse.(Float64, split(chomp(line))) for line in eachline(file)]
    end

    data = permutedims(hcat(data...))
    return data;
    
end


function sample_chain(n_samples::Int64, P_prob::Matrix{Float64}, states::Vector{Float64}) 

    chain_idx = zeros(Int64, n_samples)
    chain_states = zeros(n_samples)

    chain_idx[1] = sample(collect(1:4))
    chain_states[1] = states[chain_idx[1]]

    
    for i in 1:(n_samples-1)
        chain_idx[i+1] = sample(collect(1:4), Weights(P_prob[chain_idx[i], :]))
        chain_states[i+1] = states[chain_idx[i+1]]
    end 

    return chain_states

end




