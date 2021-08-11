import cvxpy as cp
import networkx as nx
import numpy as np


def get_communication_graph(n, p, seed):
    return nx.generators.random_graphs.binomial_graph(n=n, p=p, seed=seed)


def compute_mixing_matrix(adjacency_matrix):
    """
    computes the mixing matrix associated to a graph defined by its `adjacency_matrix` using
    FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf

    :param adjacency_matrix: np.array()
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    constraints = [
        W == W.T,
        W @ np.ones((N, 1)) == np.ones((N, 1)),
        cp.multiply(W, network_mask) == np.zeros((N, N)),
        -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
        W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N),
        np.zeros((N, N)) <= W
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    mixing_matrix = W.value

    mixing_matrix *= adjacency_matrix

    mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)

    # Force symmetry (added for numerical stability)
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :i])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix


def get_mixing_matrix(n, p, seed):
    graph = get_communication_graph(n, p, seed)
    adjacency_matrix = nx.adjacency_matrix(graph, weight=None).todense()

    return compute_mixing_matrix(adjacency_matrix)
