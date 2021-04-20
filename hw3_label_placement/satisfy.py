import typing as tp

from networkx import DiGraph

from kosaraju_sharir_scc import KosarajuSharirSCC


class TwoSATSolver:
    """
    Quadratic 2-SAT (2-satisfiability) problem solver.
    Has 2 main methods: `add_disjunction` and `solve`. First one adds
    disjunction of two variables (passed as arguments) to the
    conjunction. Second one solves the resulting conjunction.
    If the conjunction is not resolvable, `solve` returns None.

    Variables are represented with integer numbers. Absolute value represents
    the variable number, and minus sign indicates, that we take negation of the
    variable.
    """

    def __init__(self, num_variables: int) -> None:
        assert num_variables > 0
        self.implication_graph = DiGraph()
        self.implication_graph.add_nodes_from(range(1, num_variables + 1))
        self.implication_graph.add_nodes_from(range(-1, -num_variables - 1, -1))

    def add_disjunction(self, first_variable: int, second_variable: int) -> None:
        self.implication_graph.add_edge(-first_variable, second_variable)
        self.implication_graph.add_edge(-second_variable, first_variable)

    @staticmethod
    def variable_to_index(variable: int) -> int:
        return abs(variable) - 1

    @staticmethod
    def variable_to_value(variable: int) -> bool:
        return variable > 0

    @staticmethod
    def index_and_value_to_variable(variable_index: int, variable_value: bool) -> int:
        return ((-1) ** int(variable_value)) * (variable_index + 1)

    @property
    def num_variables(self) -> int:
        return self.implication_graph.number_of_nodes() // 2

    def find_graph_sccs(self) -> list[set[int]]:
        """
        Finds implication graph strongly connected components.
        """
        sccs = KosarajuSharirSCC(self.implication_graph)
        return sccs.get_sccs()

    def solve(self) -> tp.Optional[list[bool]]:
        # Find strongly connected components in resulting implication implication_graph.
        sccs = self.find_graph_sccs()

        # Check that no component contains both xi and its negation. If there is such
        # component, than formula is not satisfiable, hence, return None.
        for component in sccs:
            variables = set()
            for variable in component:
                if -variable in variables:
                    return None

                variables.add(variable)

        # Here I should have topologically sorted condenced implication_graph. However I leverage
        # the fact that SCCs come from Kosaraju-Sharir's algorithm and hence come in the
        # topological order.
        topological_order = sccs

        # Now we are ready to find a satisfying assignment.
        # We take each component in reverse topological order and set its variables to true
        # (and complementary component to false).
        result: list[tp.Optional[bool]] = [None] * self.num_variables
        for component in reversed(topological_order):
            for variable in component:
                # Note that we use notation, where variable is an integer with absolute value from
                # 1 to num_variables and its sign defines whether we need to take negation.
                variable_index = self.variable_to_index(variable)
                if result[variable_index] is not None:
                    # We found complementary component to one, that was parsed before.
                    # No additional computation needed.
                    assert all(
                        result[self.variable_to_index(variable)] is not None
                        for variable in component
                    )
                    break

                result[variable_index] = self.variable_to_value(variable)

        # mypy won't accept this kind of check and keeps complaining about the return type.
        # Hence the type ignore.
        assert all(value is not None for value in result)
        return result  # type: ignore
