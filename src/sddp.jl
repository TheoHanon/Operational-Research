using SDDP, Gurobi
using Statistics


function model_sddp(prices, states, P_prob)

    env = Gurobi.Env(output_flag = 0)
    N = length(states)
    T = length(prices)

    ub = 24 * 800 * maximum(prices)

    root = reshape(fill(1/N, N), 1, N)

    # Transition matrices: same across stages, first stage fixed at state 1
    # transition_matrices = vcat([reshape(root_transition, 1, N)], fill(P_prob, T - 1))
    transition_matrices = Vector{Matrix{Float64}}(undef, T)

    transition_matrices[1] = root
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

        #Control variables
        @variable(subproblem, 0 <= ξ <= 200)
        @variable(subproblem, 0 <= η <= 200)

        #State variable 
        @variable(subproblem, 0 <= b <= 800, SDDP.State, initial_value = 0.0)

        @constraint(subproblem, b.out == b.in + ξ - η)
        @stageobjective(subproblem, λ_t * (0.9 * η - (1 / 0.9) * ξ))
    
    end
    return model
end