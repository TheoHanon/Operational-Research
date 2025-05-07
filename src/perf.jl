include("sddp.jl")
include("deterministic.jl")
include("utils.jl")
using Statistics, Random



function compute_vss(λ̄::Vector{Float64}, ζ::Vector{Float64}, P_prob::Matrix{Float64}; N_sim::Int = 1_000)
    """
    Compute the value of the stochastic solution (VSS) for a given set of parameters.

    Parameters
    ----------
    λ̄ : Vector{Float64}
        Expected prices.
    ζ : Vector{Float64}
        Noise States.
    P_prob : Matrix{Float64}
        Transition probabilities.
    N_sddp : Int, optional
        Number of SDDP iterations. The default is 1_000.

    Returns
    -------
    vss : Float64
        Value of the stochastic solution.
    """

    m = model_sddp(λ̄, ζ, P_prob)
    SDDP.train(m; print_level=0)
    


    sims_SP = SDDP.simulate(
        m,
        N_sim;
    )
    
    profits_SP = [sum(stage[:stage_objective] for stage in sim)
                    for sim in sims_SP]
    SP = mean(profits_SP)


    
    det_ev_model = deterministic_model(λ̄)
    optimize!(det_ev_model)
    η_fix = value.(det_ev_model[:η])
    ξ_fix = value.(det_ev_model[:ξ])

    
    T = length(λ̄)
    EEV = Float64[]

    for t in 1:T
        m_t = model_sddp(λ̄, ζ, P_prob; fix_ξ = ξ_fix[1:t], fix_η = η_fix[1:t])
        SDDP.train(m_t; print_level=0)
        sims_policy = SDDP.simulate(
            m_t,
            N_sim;
        )
        push!(EEV, mean([sum(stage[:stage_objective] for stage in sim) for sim in sims_policy]))
    end

    return SP .- EEV  
end



function compute_evpi(λ̄::Vector{Float64}, ζ::Vector{Float64}, P_prob::Matrix{Float64}; N_sim::Int = 1_000)
    """
    Compute the expected value of perfect information (EVPI) for a given set of parameters.

    Parameters
    ----------
    λ̄ : Vector{Float64}
        Expected prices.
    ζ : Vector{Float64}
        Noise States.
    P_prob : Matrix{Float64}
        Transition probabilities.
    N_sim : Int, optional
        Number of simulations. The default is 1_000.
        
    Returns
    -------
    evpi : Float64
        Expected value of perfect information.

    """ 

    T = length(λ̄)
    N = length(ζ)

    
    m = model_sddp(λ̄, ζ, P_prob)
    SDDP.train(m;print_level=0)

    
    sims_SP = SDDP.simulate(
        m,
        N_sim
    )
    
    profits_SP = [sum(stage[:stage_objective] for stage in sim)
                    for sim in sims_SP]
    SP = mean(profits_SP)

    objs = Float64[]
    for _ in 1:N_sim
        ζ_path = sample_chain(T, P_prob, ζ; initial_state = 1)
        prices_path = λ̄ .* exp.(ζ_path)
        det_model = deterministic_model(prices_path)
        optimize!(det_model)
        push!(objs, objective_value(det_model))
    end

    WS = mean(objs)

    @show WS - SP

    return WS - SP

end