using JuMP, Gurobi, HiGHS

function main_problem_lshaped(p_avg, θ_init)

    m = Model(Gurobi.Optimizer)
    set_silent(m)

    @variable(m, 0 <= b1[eachindex(p_avg)] <= 800)
    @variable(m, 0 <= xi1[eachindex(p_avg)] <= 200)
    @variable(m, 0 <= eta1[eachindex(p_avg)] <= 200)
    @variable(m, θ)

    # Constraints

    @constraint(m, [t in 2:length(p_avg)], b1[t] == b1[t-1] - eta1[t]/(0.9) + 0.9 * xi1[t])
    # @constraint(m, b1[1] == 0)
    @constraint(m, θ <= θ_init)

    @objective(m, Max, sum(p_avg[t] * (eta1[t] - xi1[t]) for t in eachindex(p_avg)) + θ)

    return m, b1, eta1, xi1, θ
end

function main_problem_mc(p_avg, prob_ws, θ_init)

    m = Model(Gurobi.Optimizer)
    set_silent(m)

    @variable(m, 0 <= b1[eachindex(p_avg)] <= 800)
    @variable(m, 0 <= xi1[eachindex(p_avg)] <= 200)
    @variable(m, 0 <= eta1[eachindex(p_avg)] <= 200)
    @variable(m, θ[eachindex(prob_ws)] <= θ_init)

    # Constraints
    @constraint(m, [t in 2:length(p_avg)], b1[t] == b1[t-1] - eta1[t]/(0.9) + 0.9 * xi1[t])
    
    @objective(m, Max, sum(p_avg[t] * (eta1[t] - xi1[t]) for t in eachindex(p_avg)) + sum(θ .* prob_ws))

    return m, b1, eta1, xi1, θ
end


function solve_subproblem(b1_bar, eta1, xi1, p_w)

    m = Model(Gurobi.Optimizer)
    set_silent(m)
    @variable(m, 0 <= b2[eachindex(p_w)] <= 800)
    @variable(m, 0 <= eta2[eachindex(p_w)] <= 200)
    @variable(m, 0 <= xi2[eachindex(p_w)] <= 200)
    @variable(m, b1_end)

    # Constraints
    
    @constraint(m, b2[1] == b1_end - eta2[1] / (0.9) + 0.9 * xi2[1])
    @constraint(m, [t in 2:length(p_w)], b2[t] == b2[t-1] - eta2[t] / (0.9) + 0.9 * xi2[t])
    @constraint(m, c_link, b1_end == b1_bar[end])
    
    @objective(m, Max, sum(p_w[t] * (0.9 * eta2[t] - xi2[t]) for t in eachindex(p_w)))
    optimize!(m)

    return (obj = objective_value(m),
            π = dual(c_link), 
            b2 = value.(b2),
            eta2 = value.(eta2),
            xi2 = value.(xi2))
end


function lshaped(p_avg, p_ws, prob_ws, θ_init)

    MAXIMUM_ITERATIONS = 100
    ABSOLUTE_OPTIMALITY_GAP = 1e-6

    LBs = []
    UBs = []
    sub_ret = nothing

    # Create main model and variables
    main_model, b1, eta1, xi1, θ = main_problem_lshaped(p_avg, θ_init)

    for k in 1:MAXIMUM_ITERATIONS

        optimize!(main_model)

        if termination_status(main_model) != MOI.OPTIMAL
            error("Main problem not solved to optimality at iteration $k")
        end

        ub = objective_value(main_model)

        # Extract first-stage solution
        b1_k = value.(b1)
        eta1_k = value.(eta1)
        xi1_k = value.(xi1)

        # Solve all subproblems
        sub_ret = [
            solve_subproblem(b1_k, eta1_k, xi1_k, p_ws[l]) for l in eachindex(p_ws)
        ]

        lb = (objective_value(main_model) - value(θ)) + sum(prob_ws[i] * sub_ret[i].obj for i in eachindex(prob_ws))
        gap = abs(ub - lb) / abs(ub)

        push!(LBs, lb)
        push!(UBs, ub)

        if gap < ABSOLUTE_OPTIMALITY_GAP
            println("Terminating at iteration $(k) with the optimal solution")
            break
        end

        # Build the expected cut
        expr = 0.0
        for i in eachindex(prob_ws)
            expr += prob_ws[i] * (
                sub_ret[i].obj + (b1[end] - b1_k[end]) * sub_ret[i].π)
        end

        cut = @constraint(main_model, expr >= θ)
        @info "Adding Benders cut: $cut"

    end
    
    return (lb = LBs, ub = UBs, b1 = value.(b1), eta1 = value.(eta1), xi1 = value.(xi1), b2 = [sub_ret[i].b2 for i in eachindex(prob_ws)], eta2 = [sub_ret[i].eta2 for i in eachindex(prob_ws)], xi2 = [sub_ret[i].xi2 for i in eachindex(prob_ws)])
end


function multicut_lshaped(p_avg, p_ws, prob_ws, θ_init)

    MAXIMUM_ITERATIONS = 100
    ABSOLUTE_OPTIMALITY_GAP = 1e-6

    LBs = []
    UBs = []
    sub_ret = nothing

    # Create main model and variables
    main_model, b1, eta1, xi1, θ = main_problem_mc(p_avg, prob_ws, θ_init)


    for k in 1:MAXIMUM_ITERATIONS

        optimize!(main_model)

        if termination_status(main_model) != MOI.OPTIMAL
            error("Main problem not solved to optimality at iteration $k")
        end

        ub = objective_value(main_model)

        # Extract first-stage solution
        b1_k = value.(b1)
        eta1_k = value.(eta1)
        xi1_k = value.(xi1)

        # Solve all subproblems
        sub_ret = [
            solve_subproblem(b1_k, eta1_k, xi1_k, p_ws[l]) for l in eachindex(p_ws)
        ]

        lb = (objective_value(main_model) - sum(value.(θ) .* prob_ws)) + sum(prob_ws[i] * sub_ret[i].obj for i in eachindex(prob_ws))
        gap = abs(ub - lb) / abs(ub)

        push!(LBs, lb)
        push!(UBs, ub)

        if gap < ABSOLUTE_OPTIMALITY_GAP
            println("Terminating at iteration $(k) with the optimal solution")
            break
        end


        for i in eachindex(prob_ws)
            cut = @constraint(main_model, θ[i] <= sub_ret[i].obj + (b1[end] - b1_k[end]) * sub_ret[i].π)
            @info "Adding Benders cut: $cut"
        end

    end
    
    return (lb = LBs, ub = UBs, b1 = value.(b1), eta1 = value.(eta1), xi1 = value.(xi1), b2 = [sub_ret[i].b2 for i in eachindex(prob_ws)], eta2 = [sub_ret[i].eta2 for i in eachindex(prob_ws)], xi2 = [sub_ret[i].xi2 for i in eachindex(prob_ws)])
    
end