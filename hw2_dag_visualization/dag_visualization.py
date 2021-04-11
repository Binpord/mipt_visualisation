import networkx as nx

from coffman_graham_layering import coffman_graham_layering
from dummy_vertices_layering import dummy_vertices_layering
from crossing_reduction import reduce_crossings


def add_dummy_nodes(graph, layering):
    # Go through layers and find all edges, which go
    # through some layers. Add dummy nodes for that layers.
    dummy_nodes = set()
    new_layering = [layering[0]]
    graph = graph.copy()
    for next_layer in layering[1:]:
        new_layer = next_layer.copy()
        curr_layer = new_layering[-1]
        for node in curr_layer:
            for successor in list(graph.successors(node)):
                if successor not in next_layer:
                    graph.remove_edge(node, successor)

                    new_dummy_node = f"dummy_node{len(dummy_nodes)}"
                    graph.add_node(new_dummy_node)
                    dummy_nodes.add(new_dummy_node)
                    new_layer.add(new_dummy_node)

                    graph.add_edge(node, new_dummy_node)
                    graph.add_edge(new_dummy_node, successor)

        new_layering.append(new_layer)

    return graph, new_layering, dummy_nodes


def extract_positions(layer_ordering):
    return {
        node: (x, -y)
        for y, layer in enumerate(layer_ordering)
        for x, node in enumerate(layer)
    }


def draw(graph, max_width=None, **kwargs):
    if max_width is not None:
        layering = coffman_graham_layering(graph, max_width)
    else:
        layering = dummy_vertices_layering(graph)

    graph, layering, dummy_nodes = add_dummy_nodes(graph, layering)
    layer_ordering = reduce_crossings(graph, layering)
    positions = extract_positions(layer_ordering)
    nx.draw(
        graph,
        positions,
        nodelist=list(graph.nodes()),
        node_size=[300 if node not in dummy_nodes else 0 for node in graph.nodes()],
        **kwargs
    )
