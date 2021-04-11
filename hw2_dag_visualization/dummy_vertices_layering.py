import numpy as np

from scipy.optimize import linprog


def build_linprog_problem(graph, node_to_index):
    # f = sum(y(u) - y(v) - 1 for u, v in graph.edges)
    # -1 here is not crucial, so we skip that.
    # To emulate sum(y(u) - y(v)) with coefficients @ y,
    # I calculate resulting coefficients for every y(v) in loop.
    coefficients = np.zeros(len(graph.nodes))
    for from_node, to_node in graph.edges:
        coefficients[node_to_index[from_node]] += 1
        coefficients[node_to_index[to_node]] -= 1

    # Note that scipy.optimize.linprog only allows for constraints
    # constraints_matrix @ x <= constraints_vector
    # while in our case we want to use >=.
    # In order to fix that I negate constrain coefficients so that
    # -constraints_matrix @ x <= -constraints_vector <=> constraints_matrix @ x >= constraints_vector

    # y(v) >= 0 (or rather -y(v) <= 0)
    positivity_constraints_matrix = -np.eye(len(graph.nodes))
    positivity_constraints_vector = np.zeros(len(graph.nodes))

    # y(u) - y(v) >= 1 for all u, v in graph.edges
    # (or rather -y(u) + y(v) <= -1)
    edge_constraints_matrix = np.zeros((len(graph.edges), len(graph.nodes)))
    edge_constraints_vector = -np.ones(len(graph.edges))
    for edge_number, (from_node, to_node) in enumerate(graph.edges):
        edge_constraints_matrix[edge_number, node_to_index[from_node]] = -1
        edge_constraints_matrix[edge_number, node_to_index[to_node]] = 1

    constraints_matrix = np.concatenate(
        [positivity_constraints_matrix, edge_constraints_matrix]
    )
    constraints_vector = np.concatenate(
        [positivity_constraints_vector, edge_constraints_vector]
    )

    return coefficients, constraints_matrix, constraints_vector


def dummy_vertices_layering(graph):
    node_to_index = {node: index for index, node in enumerate(graph.nodes)}
    index_to_node = {index: node for node, index in node_to_index.items()}

    # Building and solving linear programming problem.
    # It has been proved that relaxed problem (which we are in fact solving)
    # has an integer solution. That is why, when we constraint solution
    # to be greater than or equal to 0, we should automatically get
    # integer solution.
    # Note that scipy.optimize.linprog automatically constraints solution
    # to be greater than or equal to 0.
    linprog_problem = build_linprog_problem(graph, node_to_index)
    coefficients, constraints_matrix, constraints_vector = linprog_problem
    solution = linprog(
        coefficients, constraints_matrix, constraints_vector, method="simplex"
    )
    layers_distribution = solution.x
    assert (
        all(layer_number == int(layer_number) for layer_number in layers_distribution)
        and np.min(layers_distribution) == 0
    )

    layers_distribution = layers_distribution.astype(int)
    index_to_node = {index: node for node, index in node_to_index.items()}
    layering = [
        {
            index_to_node[index]
            for index in np.where(layers_distribution == layer_number)[0]
        }
        for layer_number in range(np.max(layers_distribution) + 1)
    ]

    return list(reversed(layering))
