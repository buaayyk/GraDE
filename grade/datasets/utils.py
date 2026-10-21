import networkx
import numpy as np

def graph_hash(X ,edge_index, edge_attr, directed_flag=True, level=None):
    G = networkx.DiGraph()
    for i in range(len(X)):
        G.add_node(i, operator = X[i])
    for edge_idx in range(len(edge_index[0])):
        u = edge_index[0][edge_idx]
        v = edge_index[1][edge_idx]
        if not directed_flag or level[u] < level[v]:
            G.add_edge(u, v, edge_type=edge_attr[edge_idx])
    hash_G = networkx.weisfeiler_lehman_graph_hash(G, node_attr='operator', edge_attr='edge_type')
    return hash_G

def topological_sorted_node_indices(adj: np.array) -> np.array:
    G = networkx.DiGraph()
    for i in range(len(adj)):
        G.add_node(i)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] == 1:
                G.add_edge(i, j)
    return np.array(list(networkx.topological_sort(G)))

def topological_sorted_node_indices_from_edge_index(edge_index, n_nodes):
    G = networkx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i)
    for edge_idx in range(len(edge_index[0])):
        u = edge_index[0][edge_idx]
        v = edge_index[1][edge_idx]
        G.add_edge(u, v)
    return np.array(list(networkx.topological_sort(G)))

def alloc_node_level(adj: np.array) -> np.array:
    G = networkx.DiGraph()
    for i in range(len(adj)):
        G.add_node(i)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] == 1:
                G.add_edge(i, j)
    n_node = adj.shape[0]
    level = np.zeros((n_node))
    for v in networkx.topological_sort(G):
        for u in G.predecessors(v):
            level[v] = max(level[v], level[u] + 1)
    return level

def alloc_node_level_from_edge_index(edge_index, n_nodes):
    G = networkx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i)
    for edge_idx in range(len(edge_index[0])):
        u = edge_index[0][edge_idx]
        v = edge_index[1][edge_idx]
        G.add_edge(u, v)
    level = np.zeros((n_nodes))
    for v in networkx.topological_sort(G):
        for u in G.predecessors(v):
            level[v] = max(level[v], level[u] + 1)
    return level

def realloc_subgraph_node_level(n_node: int, sub_edge_index: np.array, full_level: np.array) -> np.array:
    G = networkx.DiGraph()
    for i in range(n_node):
        G.add_node(i)
    for edge_idx in range(sub_edge_index.shape[1]):
        u = sub_edge_index[0][edge_idx]
        v = sub_edge_index[1][edge_idx]
        if full_level[u] < full_level[v]:
            G.add_edge(u, v)
    level = np.zeros((n_node))
    for v in networkx.topological_sort(G):
        for u in G.predecessors(v):
            level[v] = max(level[v], level[u] + 1)
    return level