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
    states::Vector{Float64};
    initial_state = nothing)

    N = length(states)
    chain_idx    = Vector{Int}(undef, T)
    chain_states = Vector{Float64}(undef, T)

    # 1) uniform initial distribution over 1:N
    if initial_state !== nothing
        chain_idx[1]    = initial_state
    else
        chain_idx[1]    = sample(1:N)
    end
    chain_states[1] = states[chain_idx[1]]

    # 2) propagate
    for t in 2:T
        w = P_prob[chain_idx[t-1], :]
        chain_idx[t]    = sample(1:N, Weights(w))
        chain_states[t] = states[chain_idx[t]]
    end

    return chain_states
end

function sample_chain_idx(T::Int,
    P_prob::Matrix{Float64},
    states::Vector{Float64};
    initial_state = nothing)

    N = length(states)
    chain_idx    = Vector{Int}(undef, T)
    chain_states = Vector{Float64}(undef, T)
    chain_prob   = Vector{Float64}(undef, T)

    # 1) uniform initial distribution over 1:N
    if initial_state !== nothing
        chain_idx[1]    = initial_state
    else
        chain_idx[1]    = sample(1:N)
    end
    chain_states[1] = states[chain_idx[1]]

    # 2) propagate
    for t in 2:T
        w = P_prob[chain_idx[t-1], :]
        chain_idx[t]    = sample(1:N, Weights(w))
        chain_prob[t] = w[chain_idx[t]]
        chain_states[t] = states[chain_idx[t]]
    end

    return (chain_states, chain_prob)
end

