using JuMP, Gurobi

function deterministic_model(λ::Vector{Float64})
    
    T = length(λ)

    m = Model(Gurobi.Optimizer)
    set_silent(m)

    @variable(m, 0 <= b[1:T+1] <= 800)
    @variable(m, 0 <= ξ[1:T] <= 200)
    @variable(m, 0 <= η[1:T] <= 200)

    @constraint(m,[t in 1:T] , b[t+1] == b[t] - η[t] + ξ[t])
    @constraint(m, b[1] == 0.0)

    @objective(m, Max,
        sum(λ[t] * (0.9 * η[t] - 1.0/(0.9) * ξ[t]) for t in 1:T)
    )

    return m

end