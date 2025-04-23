using JuMP, Gurobi
using Random

# Node definition
mutable struct Node
    id::Int
    stage::Int
    parent::Union{Int, Nothing}
    prob::Float64
    state::Int
end

function build_tree(H, N, markov_transition)
    tree = Dict{Int, Node}()
    tree[1] = Node(1, 1, nothing, 1.0, 1)  # Root node: state 1

    node_id = 2
    stage_nodes = Dict(1 => [1])
    for t in 2:H
        stage_nodes[t] = []
        for parent in stage_nodes[t-1]
            for s_next in 1:N
                tree[node_id] = Node(node_id, t, parent,
                                     tree[parent].prob * markov_transition[tree[parent].state, s_next],
                                     s_next)
                push!(stage_nodes[t], node_id)
                node_id += 1
            end
        end
    end
    return tree
end

#tree = build_tree(H, N, markov_transition)

function scenario_tree(markov_support,λ_expected,tree)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, 0 <= η[n in keys(tree)] <= 200)
    @variable(model, 0 <= ξ[n in keys(tree)] <= 200)
    @variable(model, 0 <= b[n in keys(tree)] <= 800)

    @constraint(model, b[1] == 0)

    for n in values(tree)
        if !isnothing(n.parent)
            @constraint(model, b[n.id] == b[n.parent] + ξ[n.parent] - η[n.parent])
        end
    end

    λ = Dict(n.id => λ_expected[n.stage] * exp(markov_support[n.state]) for n in values(tree))

    @objective(model, Max, sum(tree[n].prob * λ[n] * (0.9 * η[n] - 1/(0.9) * ξ[n]) for n in keys(tree)))
    return model
end