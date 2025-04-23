include("sddp.jl")
include("deterministic.jl")
include("utils.jl")
using Statistics, Random



function compute_vss(λ̄::Vector{Float64}, ζ::Vector{Float64}, P_prob::Matrix{Float64}, N_sim::Int = 1_000)
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

    ############### 1) Train the multistage stochastic program ################
    sp_model = model_sddp(λ̄, ζ, P_prob)
    SDDP.train!(sp_model; iteration_limit = 250, print_level = 0)
    sims_sp  = SDDP.simulate(sp_model, N_sim, [])   # no need to save vars
    SP = mean(getfield.(sims_sp, :objective))

    ############### 2) Solve the deterministic EVP ###########################
    det_ev_model = deterministic_model(λ̄)
    optimize!(det_ev_model)
    b_var = det_ev_model[:b]
    b_hat = value.(b_var)

    ############### 3) Evaluate EVP policy under uncertainty #################
    T = length(λ̄)
    EEV_list = Vector{Float64}(undef, N_sim)

    for n in 1:N_sim
        ζ_path = sample_chain(T, P_prob, ζ)
        price_path = λ̄ .* exp.(ζ_path)

        # Build path‑specific deterministic model, but fix first‑stage variable
        pol_model = deterministic_model(price_path)
        # Assume :b is indexed 1:T
        fix(pol_model[:b][1], b_hat[1]; force = true)  # first‑stage decision
        optimize!(pol_model)
        profits_policy[n] = objective_value(pol_model)
    end

    EEV = mean(profits_policy)

    return EEV - SP   # maximise profit → VSS ≥ 0; flip sign if minimise/cost.
end



function compute_evpi(λ̄::Vector{Float64}, ζ::Vector{Float64}, P_prob::Matrix{Float64}, N_sim::Int = 1_000)
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
    SDDP.train(m;time_limit=15, print_level=0)


    # 2) Estimate SP:

    sampling_scheme = SDDP.OutOfSampleMonteCarlo(m) do node
        stage, markov_state = node          # root = (0, 0)
    
        if stage == 0
            return [SDDP.Noise((1, s), 1 / N) for s in 1:N]
    
        elseif stage == T
            children     = SDDP.Noise[]
            noise_terms  = SDDP.Noise[]       # none in this model
            return children, noise_terms
    
        else
            
            prob_row = P_prob[markov_state, :]      
            children = [SDDP.Noise((stage + 1, s), p) for (s, p) in enumerate(prob_row)]
            noise_terms = SDDP.Noise[]              # no stagewise-independent noise
            return children, noise_terms
        end
    end

    sims_SP = SDDP.simulate(
        m,
        N_sim;
        sampling_scheme = sampling_scheme,
    )
    profits_SP = [sum(stage[:stage_objective] for stage in sim)
                    for sim in sims_SP]
    SP = mean(profits_SP)

    objs = Float64[]
    for _ in 1:N_sim
        ζ_path = sample_chain(T, P_prob, ζ)
        prices_path = λ̄ .* exp.(ζ_path)
        det_model = deterministic_model(prices_path)
        optimize!(det_model)
        push!(objs, objective_value(det_model))
    end

    WS = mean(objs)

    @show WS - SP

    return WS - SP

end