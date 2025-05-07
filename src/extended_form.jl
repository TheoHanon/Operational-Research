using JuMP
using Gurobi
using Random

function generate_scenarios(H, N)
    if H == 1
        return [Int[]]
    end
    prev = generate_scenarios(H - 1, N)
    return [vcat(p, [s]) for p in prev for s in 1:N]
end


function compute_price_path(H, scenario, λ_expected, markov_support, markov_transition)
    full_states = [1; scenario]  
    λ_path = [λ_expected[t] * exp(markov_support[full_states[t]]) for t in 1:H]
    ζ_path = [full_states[t] for t in 1:H]

    prob = 1.0
    for t in 2:H
        prob *= markov_transition[full_states[t - 1], full_states[t]]
    end
    return λ_path, ζ_path, prob
end

function extended_multistage(λ_paths, ζ_paths, probs, H, num_scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, 0 <= η[1:H, 1:num_scenarios] <= 200)
    @variable(model, 0 <= ξ[1:H, 1:num_scenarios] <= 200)
    @variable(model, 0 <= b[1:H+1, 1:num_scenarios] <= 800)

    @constraint(model, b[1, :] .== 0)

    @constraint(model, [t = 1:H, ω = 1:num_scenarios],
        b[t+1, ω] == b[t, ω] + ξ[t, ω] - η[t, ω])

    for t in 1:H
        groups = Dict{NTuple{t,Int}, Vector{Int}}()  
        for ω in 1:num_scenarios
            key = Tuple(ζ_paths[ω][1:t])        
            push!( get!(groups, key, Int[]), ω )
        end
        for Ω in values(groups)                        
            ref = Ω[1]
            for ω in Ω[2:end]
                @constraint(model, η[t, ref] == η[t, ω])
                @constraint(model, ξ[t, ref] == ξ[t, ω])
            end
        end
    end

    @objective(model, Max, sum(probs[ω] * sum(λ_paths[ω][t] * (0.9*η[t, ω] - 1/(0.9) * ξ[t, ω]) for t in 1:H) for ω in 1:num_scenarios))
    return model
end