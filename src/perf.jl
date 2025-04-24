include("sddp.jl")
include("deterministic.jl")
include("utils.jl")
using Statistics, Random


# function make_sampling_scheme(m::SDDP.PolicyGraph, P_prob::Matrix{Float64})
#     """
#     Create a sampling scheme for the SDDP model.

#     Parameters
#     ----------
#     m : SDDP.MarkovianPolicyGraph
#         The SDDP model.
#     P_prob : Matrix{Float64}
#         Transition probabilities.

#     Returns
#     -------
#     sampling_scheme : function
#         The sampling scheme function.
#     """

#     T = length(P_prob)
#     N = size(P_prob, 2)

#     return SDDP.OutOfSampleMonteCarlo(m) do node
#         stage, markov_state = node          # root = (0, 0)

#         if stage == 0
#             return SDDP.Noise((1, 1), 1.0)   # root state fixed at 1
#         elseif stage == T
#             children     = SDDP.Noise[]
#             noise_terms  = SDDP.Noise[]       # none in this model
#             return children, noise_terms
#         else
#             prob_row = P_prob[markov_state, :]      
#             children = [SDDP.Noise((stage + 1, s), p) for (s, p) in enumerate(prob_row)]
#             noise_terms = SDDP.Noise[]              # no stagewise-independent noise
#             return children, noise_terms
#         end
#     end
# end


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
    # sampling_scheme = make_sampling_scheme(m, P_prob)


    # 2) Estimate SP:


    sims_SP = SDDP.simulate(
        m,
        N_sim;
        # sampling_scheme = sampling_scheme,
    )
    
    profits_SP = [sum(stage[:stage_objective] for stage in sim)
                    for sim in sims_SP]
    SP = mean(profits_SP)


    ############### 2) Solve the deterministic EVP ###########################
    det_ev_model = deterministic_model(λ̄)
    optimize!(det_ev_model)
    η_fix = value.(det_ev_model[:η])
    ξ_fix = value.(det_ev_model[:ξ])

    ############### 3) Evaluate EVP policy under uncertainty #################
    T = length(λ̄)
    EEV = Float64[]

    for t in 1:T
        m_t = model_sddp(λ̄, ζ, P_prob; fix_ξ = ξ_fix[1:t], fix_η = η_fix[1:t])
        SDDP.train(m_t; print_level=0)
        # sampling_scheme = make_sampling_scheme(m_t, P_prob)
        sims_policy = SDDP.simulate(
            m_t,
            N_sim;
            # sampling_scheme = sampling_scheme,
        )
        push!(EEV, mean([sum(stage[:stage_objective] for stage in sim) for sim in sims_policy]))
    end

    return SP .- EEV   # maximise profit → VSS ≥ 0; flip sign if minimise/cost.
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

    # 1) Train SDDP policy
    m = model_sddp(λ̄, ζ, P_prob)
    SDDP.train(m;print_level=0)

    # 2) Estimate SP:

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