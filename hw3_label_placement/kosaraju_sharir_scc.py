import networkx as nx

from depth_first_order import DepthFirstOrder


class KosarajuSharirSCC:
    """
    Finds strongly connected components in a directed graph using Kosaraju-Sharir's algorithm.

    Strongly inspired by java implementation given by Princeton's Algorithms course, available at:
        https://algs4.cs.princeton.edu/42digraph/KosarajuSharirSCC.java.html
    """

    def __init__(self, digraph: nx.DiGraph) -> None:
        self.sccs: list[set[int]] = [set()]
        self.visited: set[int] = set()

        order = DepthFirstOrder(digraph.reverse())
        for node in order.reverse_post:
            self.dfs(digraph, node)
            self.sccs.append(set())

    def dfs(self, digraph: nx.DiGraph, node: int) -> None:
        # Visit node if it was not visited.
        if node in self.visited:
            return

        self.visited.add(node)

        # Add node to the current component and visit its successors.
        self.sccs[-1].add(node)
        for successor in digraph.successors(node):
            self.dfs(digraph, successor)

    def get_sccs(self) -> list[set[int]]:
        return self.sccs
