using JuMP, Gurobi

function deterministic_model(λ::Vector{Float64})

    m = Model(Gurobi.Optimizer)
    set_silent(m)

    @variable(m, 0 <= b[1:24] <= 800)
    @variable(m, 0 <= ξ[1:24] <= 200)
    @variable(m, 0 <= η[1:24] <= 200)

    @constraint(m,[t in 2:24] ,b[t] == b[t-1] - η[t] + ξ[t])
    @constraint(m, b[1] == 0)

    @objective(m, Max,
        sum(λ[t] * (0.9 * η[t] - 1/(0.9) * ξ[t]) for t in 1:24)
    )

    return m

end