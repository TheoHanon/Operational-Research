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


function sample_chain(T::Int,
    P_prob::Matrix{Float64},
    states::Vector{Float64})

    N = length(states)
    chain_idx    = Vector{Int}(undef, T)
    chain_states = Vector{Float64}(undef, T)

    # 1) uniform initial distribution over 1:N
    init_w = fill(1/N, N)
    chain_idx[1]    = sample(1:N, Weights(init_w))
    chain_states[1] = states[chain_idx[1]]

    # 2) propagate
    for t in 2:T
        w = P_prob[chain_idx[t-1], :]
        chain_idx[t]    = sample(1:N, Weights(w))
        chain_states[t] = states[chain_idx[t]]
    end

    return chain_states
end

function sample_chain_indices(
    T::Int,
    P_prob::Matrix{Float64},
    states::Vector{Float64})

    N = length(states)
    chain_idx = Vector{Int}(undef, T)

    # 1) uniform initial distribution over 1:N
    init_w = fill(1/N, N)
    chain_idx[1] = sample(1:N, Weights(init_w))

    # 2) propagate
    for t in 2:T
        w = P_prob[chain_idx[t-1], :]
        chain_idx[t] = sample(1:N, Weights(w))
    end

    return chain_idx
end




