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

# Build full trajectories (starting from known state 1 at t=1)
function compute_price_path(scenario, λ_expected, markov_support, markov_transition)
    full_states = [1; scenario]  # root state fixed at 1
    λ_path = [λ_expected[t] * exp(markov_support[full_states[t]]) for t in 1:H]
    prob = 1.0
    for t in 2:H
        prob *= markov_transition[full_states[t - 1], full_states[t]]
    end
    return λ_path, prob
end

function extended_multistage(λ_paths, H, num_scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, 0 <= η[1:H, 1:num_scenarios] <= 200)
    @variable(model, 0 <= ξ[1:H, 1:num_scenarios] <= 200)
    @variable(model, 0 <= b[1:H+1, 1:num_scenarios] <= 800)

    #@constraint(model, [ω = 1:num_scenarios], b[1, ω] == 0)

    @constraint(model, [t = 1:H, ω = 1:num_scenarios],
        b[t+1, ω] == b[t, ω] + ξ[t, ω] - η[t, ω])

    @objective(model, Max, sum(probs[ω] * sum(λ_paths[ω][t] * (0.9*η[t, ω] - 1/(0.9) * ξ[t, ω]) for t in 1:H) for ω in 1:num_scenarios))

    optimize!(model)

    println("Objective value: ", objective_value(model))

end