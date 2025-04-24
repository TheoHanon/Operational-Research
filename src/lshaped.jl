using JuMP, Gurobi

function main_problem_lshaped(prices_expected, θ_init, F::Int)

    m = Model(Gurobi.Optimizer)
    MOI.set(m, MOI.Silent(), true)


    @variable(m, 0 <= b1[1:F+1] <= 800)
    @variable(m, 0 <= ξ1[1:F] <= 200)
    @variable(m, 0 <= η1[1:F] <= 200)
    @variable(m, θ <= θ_init)

    # Constraints

    @constraint(m, [t in 1:F], b1[t+1] == b1[t] - η1[t] + ξ1[t])
    @constraint(m, b1[1] == 0)
    
    @objective(m, Max, sum(prices_expected[t] * (0.9*η1[t] - 1/(0.9) * ξ1[t]) for t in 1:F) + θ)

    return m, b1, η1, ξ1, θ
end

function main_problem_mc(prices_expected, prices_prob, θ_init, F::Int)

    m = Model(Gurobi.Optimizer)
    MOI.set(m, MOI.Silent(), true)


    @variable(m, 0 <= b1[1:F+1] <= 800)
    @variable(m, 0 <= ξ1[1:F] <= 200)
    @variable(m, 0 <= η1[1:F] <= 200)
    @variable(m, θ[eachindex(prices_prob)] <= θ_init)

    # Constraints
    @constraint(m, [t in 1:F], b1[t+1] == b1[t] - η1[t] + ξ1[t])
    @constraint(m, b1[1] == 0)
    
    @objective(m, Max, sum(prices_expected[t] * (0.9*η1[t] - 1/(0.9) * ξ1[t]) for t in 1:F) + sum(θ .* prices_prob))

    return m, b1, η1, ξ1, θ
end


function solve_subproblem(b1, η1, ξ1, prices_scenarios, F::Int)

    m = Model(Gurobi.Optimizer)
    MOI.set(m, MOI.Silent(), true)

    @variable(m, 0 <= b2[1:24-F + 1] <= 800)
    @variable(m, 0 <= η2[1:24-F] <= 200)
    @variable(m, 0 <= ξ2[1:24-F] <= 200)
    @variable(m, b1_link)

    # Constraints
    
    @constraint(m, b2[1] == b1_link)
    @constraint(m, [t in 1:(24-F)], b2[t+1] == b2[t] - η2[t] + ξ2[t])
    @constraint(m, c_link, b1_link ==  b1[end])
    
    @objective(m, Max, sum(prices_scenarios[t] * (0.9 * η2[t] - 1/(0.9) * ξ2[t]) for t in 1:(24-F)))
    optimize!(m)

    return (obj = objective_value(m),
            π = dual(c_link), 
            b2 = value.(b2),
            η2 = value.(η2),
            ξ2 = value.(ξ2))
end


function lshaped(prices_expected, prices_scenarios, prices_prob, θ_init)

    MAXIMUM_ITERATIONS = 100
    ABSOLUTE_OPTIMALITY_GAP = 1e-12
    F = length(prices_expected)

    LBs = []
    UBs = []
    sub_ret = nothing

    # Create main model and variables
    main_model, b1, η1, ξ1, θ = main_problem_lshaped(prices_expected, θ_init, F)

    for k in 1:MAXIMUM_ITERATIONS

        optimize!(main_model)

        if termination_status(main_model) != MOI.OPTIMAL
            error("Main problem not solved to optimality at iteration $k")
        end

        ub = objective_value(main_model)

        # Extract first-stage solution
        b1_k = value.(b1)
        eta1_k = value.(η1)
        xi1_k = value.(ξ1)

        # Solve all subproblems
        sub_ret = [
            solve_subproblem(b1_k, eta1_k, xi1_k, prices_scenarios[l], F) for l in eachindex(prices_scenarios)
        ]

        lb = (objective_value(main_model) - value(θ)) + sum(prices_prob[i] * sub_ret[i].obj for i in eachindex(prices_scenarios))
        gap = abs(ub - lb) / abs(ub)

        push!(LBs, lb)
        push!(UBs, ub)

        if gap < ABSOLUTE_OPTIMALITY_GAP
            println("Terminating at iteration $(k) with the optimal solution")
            break
        end

        # Build the expected cut
        expr = 0.0
        for i in eachindex(prices_prob)
            expr += prices_prob[i] * (
                sub_ret[i].obj + (b1[end] - b1_k[end]) * sub_ret[i].π)
        end

        cut = @constraint(main_model, expr >= θ)
        @info "Adding Benders cut: $cut"

    end
    
    return (
        lb = LBs, 
        ub = UBs, 
        b1 = value.(b1), 
        η1 = value.(η1), 
        ξ1 = value.(ξ1), 
        b2 = [r.b2  for r in sub_ret], 
        η2 = [r.η2 for r in sub_ret],
        ξ2 = [r.ξ2 for r in sub_ret]
    )
end


function multicut_lshaped(prices_expected, prices_scenarios, prices_prob, θ_init)

    MAXIMUM_ITERATIONS = 100
    ABSOLUTE_OPTIMALITY_GAP = 1e-6
    F = length(prices_expected)

    LBs = []
    UBs = []
    sub_ret = nothing

    # Create main model and variables
    main_model, b1, η1, ξ1, θ = main_problem_mc(prices_expected, prices_prob, θ_init, F)

    for k in 1:MAXIMUM_ITERATIONS

        optimize!(main_model)

        if termination_status(main_model) != MOI.OPTIMAL
            error("Main problem not solved to optimality at iteration $k")
        end

        ub = objective_value(main_model)

        # Extract first-stage solution
        b1_k = value.(b1)
        eta1_k = value.(η1)
        xi1_k = value.(ξ1)

        # Solve all subproblems
        sub_ret = [
            solve_subproblem(b1_k, eta1_k, xi1_k, prices_scenarios[l], F) for l in eachindex(prices_scenarios)
        ]

        lb = (objective_value(main_model) - sum(value.(θ) .* prices_prob)) + sum(prices_prob[i] * sub_ret[i].obj for i in eachindex(prices_scenarios))
        gap = abs(ub - lb) / abs(ub)

        push!(LBs, lb)
        push!(UBs, ub)

        if gap < ABSOLUTE_OPTIMALITY_GAP
            println("Terminating at iteration $(k) with the optimal solution")
            break
        end


        for i in eachindex(prices_scenarios)
            cut = @constraint(main_model, θ[i] <= sub_ret[i].obj + (b1[end] - b1_k[end]) * sub_ret[i].π)
            @info "Adding Benders cut: $cut"
        end

    end
    
    return (
        lb = LBs, 
        ub = UBs, 
        b1 = value.(b1), 
        η1 = value.(η1), 
        ξ1 = value.(ξ1), 
        b2 = [r.b2  for r in sub_ret], 
        η2 = [r.η2 for r in sub_ret],
        ξ2 = [r.ξ2 for r in sub_ret]
    )
    
end