import typing as tp

import networkx as nx


class DepthFirstOrder:
    """
    Orders nodes in a directed graph in (reversed) postorder.

    Strongly inspired by java implementation given by Princeton's Algorithms course, available at:
        https://algs4.cs.princeton.edu/42digraph/DepthFirstOrder.java.html
    """

    def __init__(self, digraph: nx.DiGraph) -> None:
        self.postorder: list[int] = []
        self.visited: set[int] = set()
        for node in digraph.nodes():
            self.dfs(digraph, node)

    def dfs(self, digraph: nx.DiGraph, node: int) -> None:
        if node in self.visited:
            return

        self.visited.add(node)

        for successor in digraph.successors(node):
            self.dfs(digraph, successor)

        self.postorder.append(node)

    @property
    def reverse_post(self) -> tp.Iterable[int]:
        return reversed(self.postorder)
