import networkx as nx

from coffman_graham_layering import coffman_graham_layering
from dummy_vertices_layering import dummy_vertices_layering
from crossing_reduction import reduce_crossings


def extract_positions(layer_ordering):
    return {
        node: (x, y)
        for y, layer in enumerate(layer_ordering)
        for x, node in enumerate(layer)
    }


def draw(graph, max_width=None, **kwargs):
    if max_width is not None:
        layering = coffman_graham_layering(graph, max_width)
    else:
        layering = dummy_vertices_layering(graph)

    layer_ordering = reduce_crossings(graph, layering)
    positions = extract_positions(layer_ordering)
    nx.draw(graph, positions, **kwargs)
