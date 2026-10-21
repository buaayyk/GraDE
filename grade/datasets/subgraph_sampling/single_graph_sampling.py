import networkx
import random
from collections import defaultdict

def neighbour_reservoir_sampling(G: networkx.DiGraph, k: int):
    """
    Neighbour Reservoir Sampling (NRS) algorithm for sampling connected induced subgraphs uniformly at random.
    
    Parameters:
    G (networkx.DiGraph): The input graph (must be connected)
    k (int): Size of the subgraph to sample
    
    Returns:
    networkx.DiGraph: A connected induced subgraph with k vertices
    """
    # Step 0: Keep only one direction in undirected graph
    directed_flag = False
    for u, v in G.edges():
        if (v, u) not in G.edges():
            directed_flag = True
            break
    if not directed_flag:
        G_processed = networkx.DiGraph()
        G_processed.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if u <= v:
                G_processed.add_edge(u, v)
        G = G_processed

    # Step 1: Initial selection using RVE (Random Vertex Expansion)
    # Start from a random edge
    edges = list(G.edges())
    u, v = random.choice(edges)
    V = {u, v}
    edge_dict = defaultdict(dict) # {v: {(u, v): idx}}
    EL_list = [] # List of (u, v)

    # Helper to add edge to EL
    def add_edge(u, v):
        if v not in V:
            idx = len(EL_list)
            EL_list.append((u, v))
            edge_dict[v][u, v] = idx
    
    def remove_edge(u, v):
        if (u, v) in edge_dict[v]:
            idx = edge_dict[v][(u, v)]
            # Swap with the last element
            last_u, last_v = EL_list[-1]
            EL_list[idx] = (last_u, last_v)
            # Update the moved edge's index
            edge_dict[last_v][last_u, last_v] = idx
            # Remove from EL_list and edge_dict
            EL_list.pop()
            del edge_dict[v][u, v]
    
    # Initialize EL
    for u in V:
        for v in G.successors(u):
            add_edge(u, v)
        for v in G.predecessors(u):
            add_edge(u, v) # Store as (u, v)
    
    # Expand to k vertices
    while len(V) < k and EL_list:
        # Randomly select an edge
        idx = random.randint(0, len(EL_list) - 1)
        u, v = EL_list[idx]

        V.add(v)

        # Remove selected edge
        remove_edge(u, v)
        
        # Remove all edges between v and existing V nodes
        for (w, x) in list(edge_dict.get(v, {}).keys()):
            remove_edge(w, x) # x == v

        # Add new edge from v
        for w in G.successors(v):
            if w not in V:
                add_edge(v, w)
        for w in G.predecessors(v):
            if w not in V:
                add_edge(v, w) # Store ad (v, w)

    # Step 2: Reservoir sampling phase
    i = k
    unprocessed = set(G.nodes()) - V
    edge_dict = defaultdict(dict)
    EL_list = []

    # Initialize EL for reservoir phase
    for u in V:
        for v in G.successors(u):
            if v in unprocessed:
                add_edge(u, v)
        for v in G.predecessors(u):
            if v in unprocessed:
                add_edge(u, v) # Store as (u, v)

    while EL_list and unprocessed:
        # Randomly select an edge
        idx = random.randint(0, len(EL_list) - 1)
        u, v = EL_list[idx]
        i += 1
        alpha = random.random()

        unprocessed.remove(v)
        # Remove all edges between v and processed nodes
        for (w, x) in list(edge_dict.get(v, {}).keys()):
            remove_edge(w, x) # x == v

        if alpha < k / i:
            # Replace a random node in V with v
            node_to_replace = random.choice(list(V))
            V_candidate = V - {node_to_replace} | {v}

            # Check weak connectivity for directed graphs
            if networkx.weakly_connected.is_weakly_connected(G.subgraph(V_candidate)):
                # Remove all edges connected to node_to_replace
                for (w, x) in list(edge_dict.get(node_to_replace, {}).keys()):
                    remove_edge(w, x) # x == node_to_place

                # Add new edges from v
                for w in G.successors(v):
                    if w in unprocessed:
                        add_edge(v, w)
                for w in G.predecessors(v):
                    if w in unprocessed:
                        add_edge(v, w)

                # Update V
                V = V_candidate
    
    return G.subgraph(V)