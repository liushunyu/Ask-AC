import numpy as np
from gymnasium.spaces.space import Space


class Graph(Space):
    r"""A graph space.
    Example::
        >>> Graph(2,2)
    """
    def __init__(self, node_num, feature_dim):
        assert node_num >= 0
        self.node_num = node_num
        self.feature_dim = feature_dim
        super(Graph, self).__init__((), np.int64)

    def __repr__(self):
        return "Graph(%d)" % self.node_num

    def __eq__(self, other):
        return isinstance(other, Graph) and self.node_num == other.node_num and self.feature_dim == other.feature_dim
