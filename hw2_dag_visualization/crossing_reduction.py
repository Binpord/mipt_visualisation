import itertools

import numpy as np


def tripletwise(iterable):
    # Based on pairwise from itertools documentation.
    a, b, c = itertools.tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


def common_nodes_indices(nodes, layer):
    return [layer.index(node) for node in nodes if node in layer]


def count_crossings(prev_layer, middle_layer, next_layer, graph):
    crossings_count = 0

    # Cache predecessors and successors indices.
    predecessors_indices = {
        node: np.array(common_nodes_indices(graph.predecessors(node), prev_layer))
        for node in middle_layer
    }
    successors_indices = {
        node: np.array(common_nodes_indices(graph.successors(node), next_layer))
        for node in middle_layer
    }

    # Iterate over all pairs of nodes in middle layer and count
    # number of crossings between edges from their predecessors and
    # to their successors.
    # Note that itertools.combinations yields nodes in correct order,
    # i.e. first will always be the left one and second - right.
    for left_node, right_node in itertools.combinations(middle_layer, 2):
        for predecessor_index in predecessors_indices[left_node]:
            crossings_count += np.sum(
                predecessors_indices[right_node] < predecessor_index
            )

        for successor_index in successors_indices[left_node]:
            crossings_count += np.sum(successors_indices[right_node] < successor_index)

    return crossings_count


def local_search(layer_ordering, graph, reversed_order=False):
    # Local search preserves ordering of first and last layers.
    new_ordering = [layer_ordering[0] if not reversed_order else layer_ordering[-1]]

    # Iterate over triplets of layers and choose middle layer permutation
    # based on number of crossings of edges between layers.
    triplets = tripletwise(
        layer_ordering if not reversed_order else reversed(layer_ordering)
    )
    for prev_layer, curr_layer, next_layer in triplets:
        _, best_permutation = min(
            (
                count_crossings(prev_layer, permutation, next_layer, graph),
                permutation,
            )
            for permutation in itertools.permutations(curr_layer)
        )
        new_ordering.append(best_permutation)

    # Local search preserves ordering of first and last layers.
    new_ordering.append(layer_ordering[-1] if not reversed_order else layer_ordering[0])
    return new_ordering if not reversed_order else list(reversed(new_ordering))


def median_layer_ordering(layering, graph):
    # Choose arbitrary first layer ordering
    layer_ordering = [list(layering[0])]

    # For every consequent layer choose ordering based
    # on median of nodes' predecessors in previous layer.
    for new_layer in layering[1:]:
        prev_layer = layer_ordering[-1]
        predecessors_indices = {
            node: common_nodes_indices(graph.predecessors(node), prev_layer)
            for node in new_layer
        }
        predecessors_medians = {
            node: np.median(indices) if indices else float("inf")
            for node, indices in predecessors_indices.items()
        }
        layer_ordering.append(
            list(sorted(new_layer, key=lambda node: predecessors_medians[node]))
        )

    return layer_ordering


def reduce_crossings(graph, layering, n_sweeps=2):
    # Choose initial layer ordering using median method.
    layer_ordering = median_layer_ordering(layering, graph)

    # Do several local search sweeps.
    for _ in range(n_sweeps):
        layer_ordering = local_search(layer_ordering, graph, False)
        layer_ordering = local_search(layer_ordering, graph, True)

    return layer_ordering
