import functools


# Auxiliary class, which implements specific order on finite sets of positive integers,
# used by Coffman-Graham-Layering algorithm.
@functools.total_ordering
class OrderedSet:
    def __init__(self, elements):
        self.elements = list(elements)
        self.max = max(self.elements, default=None)

    def __lt__(self, other):
        if not isinstance(other, OrderedSet):
            raise ValueError(
                f"unexpected other of type {type(other)} in OrderedSet.__lt__"
            )

        # Order is defined as follows. self < other if either:
        # 1. self is empty and other is not empty, or
        # 2. self and other both aren't empty and max(self) < max(other), or
        # 3. self and other both aren't empty, max(self) = max(other)
        #    and {self without max(self)} < {other without max(other)}

        if self.is_empty():
            return not other.is_empty()
        elif other.is_empty():
            return False

        # here self and other both aren't empty
        if self.max < other.max:
            return True

        return self.remove_max() < other.remove_max()

    def is_empty(self):
        return len(self.elements) == 0

    def remove_max(self):
        # Makes new set {self without max(self)}
        return OrderedSet(element for element in self.elements if element != self.max)


def label_nodes(graph):
    # Initially, all nodes are unlabeled.
    inf = float("inf")
    labels = {node: inf for node in graph.nodes}

    for label in range(len(graph.nodes)):
        # Choose an unlabeled node, which minimizes its predecessors' set.
        _, node = min(
            (
                OrderedSet(
                    labels[predecessor] for predecessor in graph.predecessors(node)
                ),
                node,
            )
            for node in graph.nodes
            if labels[node] == inf
        )
        labels[node] = label

    return labels


def assign_layers(graph, labels, max_width):
    # Assign nodes to layers based on labels.
    # Result is Coffman-Graham layering, which is list of sets, each
    # of which contains nodes, assigned to corresponding layers.
    graph_nodes = set(graph.nodes)
    used_nodes = set()
    layering = [set()]

    for _ in range(len(graph.nodes)):
        # Choose unassigned node, such that all its successors are already assigned and
        # label of this vertex is maximized.
        _, node = max(
            (labels[node], node)
            for node in graph_nodes - used_nodes
            if all(successor in used_nodes for successor in graph.successors(node))
        )

        # Now we should assign selected node to a layer.
        # We want to assign it to current layer (as our main goal is to minimize overall
        # amount of layers), but it is not possible if either current layer is full
        # (external constraint on maximal layer width) or any of current node successors
        # is already on this layer (we don't want edges on final image to be horizontal).
        # In both mentioned cases we simply close current layer and open new one.
        current_layer = layering[-1]
        node_successors = graph.successors(node)
        if len(current_layer) == max_width or any(
            successor in current_layer for successor in node_successors
        ):
            layering.append(set())

        # Add node to last layer (which is either current layer or a new layer, which
        # we just opened) and mark node as assigned.
        layering[-1].add(node)
        used_nodes.add(node)

    return list(reversed(layering))


def coffman_graham_layering(graph, max_width):
    labels = label_nodes(graph)
    return assign_layers(graph, labels, max_width)
