using SDDP, Gurobi
using Statistics


function model_sddp(prices, states, P_prob;
    fix_ξ = nothing,
    fix_η = nothing)

    env = Gurobi.Env()
    N = length(states)
    T = length(prices)

    ub = 24 * 200 * maximum(prices)

    root_transition = zeros(1, N)
    root_transition[1] = 1.0
    transition_matrices = Vector{Matrix{Float64}}(undef, T)

    transition_matrices[1] = reshape(root_transition, 1, N)
    for t in 2:T
        transition_matrices[t] = P_prob
    end
    
    model = SDDP.MarkovianPolicyGraph(
        transition_matrices = transition_matrices,
        sense = :Max,
        upper_bound = ub,
        optimizer = () -> Gurobi.Optimizer(env)
    ) do subproblem, node

        stage, markov_state = node
        λ_t = prices[stage] * exp(states[markov_state])

        
        @variable(subproblem, 0 <= ξ <= 200)
        @variable(subproblem, 0 <= η <= 200)

        
        @variable(subproblem, 0 <= b <= 800, SDDP.State, initial_value = 0.0)

        @constraint(subproblem, b.out == b.in + ξ - η)

        if (fix_ξ !== nothing && stage <= length(fix_ξ)) && (fix_η !== nothing && stage <= length(fix_η))
            @constraint(subproblem, ξ == fix_ξ[stage])
            @constraint(subproblem, η == fix_η[stage])
        end

        @stageobjective(subproblem, λ_t * (0.9 * η - (1.0 / 0.9) * ξ))
    
    end
    return model
end