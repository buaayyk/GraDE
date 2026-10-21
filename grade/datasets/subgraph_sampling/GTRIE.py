import collections
import itertools
import random
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch_geometric
import tqdm
from grade.datasets.utils import graph_hash, realloc_subgraph_node_level

try:
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_subgraph_hashes

    _HAS_WL = True
except Exception:
    _HAS_WL = False

NO_EDGE = -1

# node feature -> discrete label, g-trie matching.
def _discrete_node_labels(x: torch.Tensor) -> List[int]:
    if x.dim() == 2:
        return torch.argmax(x, dim=-1).cpu().tolist()
    return x.cpu().tolist()

# edge feature -> discrete_type id. Both for one-hot & int features.
def _discrete_edge_types(edge_attr: torch.Tensor | None) -> List[int] | None:
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        return edge_attr.long().cpu().tolist()
    return torch.argmax(edge_attr, dim=1).cpu().tolist()

class GTrieNode:
    def __init__(
        self,
        depth: int,
        node_label: int | None = None,
        edge_constraints: Tuple[Tuple[int, int], ...] = tuple(),
    ):
        self.depth = depth                         
        self.node_label = node_label                
        self.edge_constraints = edge_constraints    
        self.children: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], "GTrieNode"] = {} 
        self.is_leaf = False
        self.pattern_hash: str | None = None

class GTrie:
    def __init__(self, directed_flag: bool):
        self.directed_flag = directed_flag
        self.root = GTrieNode(depth=0)

    # insert pattern's signature into trie, leaf record pattern_hash.
    def insert(self, signature: Sequence[Tuple[int, Tuple[Tuple[int, int], ...]]], pattern_hash: str):
        node = self.root 
        for node_label, edges in signature: 
            key = (node_label, edges)
            if key not in node.children:
                node.children[key] = GTrieNode(depth=node.depth + 1, node_label=node_label, edge_constraints=edges)
            node = node.children[key]
        node.is_leaf = True 
        node.pattern_hash = pattern_hash


    def has_patterns(self) -> bool:
        return len(self.root.children) > 0

    # host graph(adj_matrix + node_labels)
    def match(self, adj_matrix: List[List[int]], node_labels: List[int]) -> Iterable[Tuple[str, List[int]]]:
        n_node = len(node_labels)
        label_to_nodes: Dict[int, List[int]] = collections.defaultdict(list)
        for idx, label in enumerate(node_labels):
            label_to_nodes[label].append(idx)
        used = [False] * n_node
        mapping: List[int] = []
        seen_subsets: set[Tuple[int, ...]] = set()

        # internal recursive function
        def dfs(node: GTrieNode):
            
            if node.is_leaf and node.pattern_hash is not None:
                key = tuple(sorted(mapping))
                if key not in seen_subsets:
                    seen_subsets.add(key)
                    yield node.pattern_hash, list(mapping)
                return
            for child in node.children.values():
                for candidate in self._candidate_vertices(child, mapping, used, label_to_nodes, adj_matrix):
                    used[candidate] = True
                    mapping.append(candidate)
                    yield from dfs(child)
                    mapping.pop()
                    used[candidate] = False

        yield from dfs(self.root)

    def sample_matches(self, adj_matrix: List[List[int]], node_labels: List[int], p_list: Sequence[float], rng: random.Random, max_walks: int = 1000) -> Iterable[Tuple[List[int], float]]:
        assert len(p_list) > 0
        n_node = len(node_labels)
        label_to_nodes: Dict[int, List[int]] = collections.defaultdict(list)
        for idx, label in enumerate(node_labels):
            label_to_nodes[label].append(idx)

        def walk():
            used = [False] * n_node
            mapping: List[int] = []

            def dfs(node: GTrieNode, depth: int, current_prob: float):
                if depth > 0:
                    if depth - 1 >= len(p_list):
                        return None
                    cont_prob = p_list[depth - 1]
                    if cont_prob <= 0:
                        return None
                    if rng.random() > cont_prob:
                        return None
                    current_prob *= cont_prob
                if node.is_leaf and node.pattern_hash is not None:
                    return list(mapping), current_prob
                options = []
                for child in node.children.values():
                    candidates = list(self._candidate_vertices(child, mapping, used, label_to_nodes, adj_matrix))
                    if candidates:
                        options.append((child, candidates))
                if not options:
                    return None
                child, candidates = options[rng.randrange(len(options))]
                current_prob *= 1.0 / len(options)
                candidate = candidates[rng.randrange(len(candidates))]
                current_prob *= 1.0 / len(candidates)
                used[candidate] = True
                mapping.append(candidate)
                result = dfs(child, child.depth, current_prob)
                mapping.pop()
                used[candidate] = False
                return result

            return dfs(self.root, self.root.depth, 1.0)

        if not self.root.children:
            return
        attempts = 0
        while attempts < max_walks:
            attempts += 1
            res = walk()
            if res is None:
                continue
            yield res

    def _candidate_vertices(self, child: GTrieNode, mapping: Sequence[int], used: List[bool], label_to_nodes: Dict[int, List[int]], adj_matrix: List[List[int]]) -> Iterable[int]:
        candidates = label_to_nodes.get(child.node_label, [])
        if not candidates:
            return []
        for vertex in candidates:
            if used[vertex]:
                continue
            if self._compatible(vertex, mapping, child.edge_constraints, adj_matrix):
                yield vertex

    def _compatible(self, vertex: int, mapping: Sequence[int], constraints: Tuple[Tuple[int, int], ...], adj_matrix: List[List[int]]) -> bool:
        for idx, mapped_vertex in enumerate(mapping):
            required_out, required_in = constraints[idx]
            actual_out = adj_matrix[vertex][mapped_vertex]
            actual_in = adj_matrix[mapped_vertex][vertex]
            if required_out != actual_out or required_in != actual_in:
                return False
        return True

def build_graph_representation(datapoint, directed_flag: bool) -> Tuple[List[List[int]], List[set], List[int]]:
    n_node = datapoint.x.shape[0]
    adj_matrix = [[NO_EDGE for _ in range(n_node)] for _ in range(n_node)]
    neighbors = [set() for _ in range(n_node)]
    node_labels = _discrete_node_labels(datapoint.x)
    edge_index = datapoint.edge_index.cpu().tolist()
    edge_types = _discrete_edge_types(datapoint.edge_attr)
    if directed_flag:
        level = datapoint.level.cpu().tolist()
    else:
        level = None
    for edge_idx in range(len(edge_index[0])):
        u = edge_index[0][edge_idx]
        v = edge_index[1][edge_idx]
        attr = edge_types[edge_idx] if edge_types is not None else 0
        if directed_flag:
            if int(level[u]) < int(level[v]):
                adj_matrix[u][v] = attr
                neighbors[u].add(v)
                neighbors[v].add(u)
        else:
            adj_matrix[u][v] = attr
            adj_matrix[v][u] = attr
            neighbors[u].add(v)
            neighbors[v].add(u)
    return adj_matrix, neighbors, node_labels


def enumerate_connected_subsets(neighbors: List[set], subgraph_size: int) -> Iterable[List[int]]:
    n_node = len(neighbors)
    if subgraph_size == 1:
        for node in range(n_node):
            yield [node]
        return

    def extend(current: set, extension: set, root: int):
        if len(current) == subgraph_size:
            yield list(current)
            return
        extension_local = set(extension)
        while extension_local:
            w = extension_local.pop()
            new_extension = set(extension_local)
            for neighbor in neighbors[w]:
                if neighbor > root and neighbor not in current:
                    new_extension.add(neighbor)
            yield from extend(current | {w}, new_extension, root)

    for root in range(n_node):
        extension = {neighbor for neighbor in neighbors[root] if neighbor > root}
        yield from extend({root}, extension, root)


def _induced_adj_matrix(adj_matrix: List[List[int]], subset: Sequence[int]) -> List[List[int]]:
    return [[adj_matrix[u][v] for v in subset] for u in subset]


def build_signature(subset: Sequence[int], adj_matrix: List[List[int]], node_labels: List[int], directed_flag: bool) -> List[Tuple[int, Tuple[Tuple[int, int], ...]]]:
    if not subset:
        return []
    induced = _induced_adj_matrix(adj_matrix, subset)
    order = _canonical_node_order(subset, induced, node_labels, directed_flag)
    signature: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
    for depth, idx in enumerate(order):
        label = node_labels[subset[idx]]
        if depth == 0:
            signature.append((label, tuple()))
            continue
        edges = []
        for prev_idx in order[:depth]:
            prev = subset[prev_idx]
            cur = subset[idx]
            out_type = adj_matrix[cur][prev]
            in_type = adj_matrix[prev][cur]
            if directed_flag:
                edges.append((out_type, in_type))
            else:
                edges.append((out_type, out_type))
        signature.append((label, tuple(edges)))
    return signature


def _canonical_node_order(subset: Sequence[int], induced: List[List[int]], node_labels: List[int], directed_flag: bool) -> List[int]:
    n = len(subset)
    if _HAS_WL:
        G = nx.DiGraph() if directed_flag else nx.Graph()
        for new_idx, orig_idx in enumerate(subset):
            G.add_node(new_idx, label=node_labels[orig_idx])
        for i in range(n):
            for j in range(n):
                edge_type = induced[i][j]
                if edge_type != NO_EDGE:
                    G.add_edge(i, j, edge_type=edge_type)
        wl_hashes = weisfeiler_lehman_subgraph_hashes(
            G,
            node_attr="label",
            edge_attr="edge_type",
            iterations=max(2, n),
        )
        return sorted(range(n), key=lambda idx: (wl_hashes[idx], node_labels[subset[idx]], idx))
    # Fallback: degree-based deterministic order
    order_keys = []
    for i in range(n):
        neighbors_info = []
        for j in range(n):
            out_edge = induced[i][j]
            in_edge = induced[j][i]
            if out_edge != NO_EDGE or in_edge != NO_EDGE:
                neighbors_info.append((node_labels[subset[j]], out_edge, in_edge))
        neighbors_info.sort()
        order_keys.append((node_labels[subset[i]], tuple(neighbors_info), i))
    return [entry[2] for entry in sorted(order_keys)]


class GraphRepresentationCache:
    def __init__(self, data: Sequence, directed_flag: bool, max_size: int = 64):
        self.data = data
        self.directed_flag = directed_flag
        self.max_size = max_size
        self.cache: "OrderedDict[int, Tuple[List[List[int]], List[int]]]" = OrderedDict()

    def get(self, idx: int) -> Tuple[List[List[int]], List[int]]:
        if self.max_size <= 0:
            adj_matrix, _, node_labels = build_graph_representation(self.data[idx], self.directed_flag)
            return adj_matrix, node_labels
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        adj_matrix, _, node_labels = build_graph_representation(self.data[idx], self.directed_flag)
        self.cache[idx] = (adj_matrix, node_labels)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return self.cache[idx]

def extract_subgraph(datapoint, subset: List[int], directed_flag: bool, materialize_data: bool = True):

    sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
        subset=subset,
        edge_index=datapoint.edge_index,
        edge_attr=datapoint.edge_attr,
        relabel_nodes=True,
        num_nodes=datapoint.x.shape[0],
    )
    sub_x = datapoint.x[subset]
    sub_y = datapoint.y
    if directed_flag:
        sub_edge_index_np = sub_edge_index.cpu().numpy()
        level_np = datapoint.level[subset].cpu().numpy()
        sub_level = realloc_subgraph_node_level(len(subset), sub_edge_index_np, level_np)
        sub_level_np = np.array(sub_level, dtype=float)
    else:
        sub_level_np = None
    if sub_x.dim() == 2:
        sub_X = torch.argmax(sub_x, dim=-1).cpu().tolist()
    else:
        sub_X = sub_x.cpu().tolist()
    sub_edge_index_list = sub_edge_index.cpu().tolist()
    sub_edge_attr_idx = _discrete_edge_types(sub_edge_attr)
    if sub_edge_attr_idx is None:
        sub_edge_attr_idx = [0 for _ in range(sub_edge_index.shape[1])]
    if directed_flag:
        hash_G = graph_hash(sub_X, sub_edge_index_list, sub_edge_attr_idx, True, sub_level_np)
    else:
        hash_G = graph_hash(sub_X, sub_edge_index_list, sub_edge_attr_idx, False, None)
    if not materialize_data:
        return None, hash_G
    if directed_flag:
        sub_level_tensor = torch.tensor(sub_level_np, dtype=torch.float32, device=sub_x.device)
        subgraph_datapoint = torch_geometric.data.Data(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            y=sub_y,
            level=sub_level_tensor,
        )
    else:
        subgraph_datapoint = torch_geometric.data.Data(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            y=sub_y,
        )
    return subgraph_datapoint, hash_G

def generate_all_non_isomorphic_connected_graphs(k: int, directed_flag: bool) -> List[nx.Graph]:
    """
    Generate all non-isomorphic connected graphs with k nodes.
    For directed graphs, generate directed graphs.
    """
    if directed_flag:
        # For directed: 2^(k*(k-1)) possibilities (no self-loops)
        possible_edges = list(itertools.product(range(k), repeat=2))
        possible_edges = [(u, v) for u, v in possible_edges if u != v]  # No self-loops
    else:
        # For undirected: 2^(k*(k-1)/2) possibilities
        possible_edges = [(u, v) for u in range(k) for v in range(u+1, k)]
    
    unique_graphs = []
    seen_hashes = set()
    
    for edge_subset in itertools.product([0, 1], repeat=len(possible_edges)):
        G = nx.DiGraph() if directed_flag else nx.Graph()
        G.add_nodes_from(range(k))
        for i, (u, v) in enumerate(possible_edges):
            if edge_subset[i]:
                G.add_edge(u, v)
        
        # Check connectivity
        if not nx.is_connected(G) if not directed_flag else not nx.is_weakly_connected(G):
            continue
        
        # Canonical hash using WL (if available)
        if _HAS_WL:
            wl_hashes = weisfeiler_lehman_subgraph_hashes(G, iterations=max(2, k))
            hash_val = tuple(sorted(wl_hashes.values()))
        else:
            # Fallback: adjacency matrix hash
            adj = nx.to_numpy_array(G, dtype=int)
            hash_val = tuple(adj.flatten())
        
        if hash_val not in seen_hashes:
            seen_hashes.add(hash_val)
            unique_graphs.append(G)
    
    return unique_graphs

def construct_gtrie(subgraph_size: int, directed_flag: bool):
    gtrie = GTrie(directed_flag=directed_flag)
    # Generate all non-isomorphic connected graphs
    all_graphs = generate_all_non_isomorphic_connected_graphs(subgraph_size, directed_flag)
    
    for G in tqdm.tqdm(all_graphs, desc="building g-trie"):
        # Convert NetworkX graph to adj_matrix and node_labels
        adj_matrix = nx.to_numpy_array(G, dtype=int).tolist()
        node_labels = [0] * subgraph_size  # Assume all nodes have label 0
        
        # Build signature for the full graph (subset is all nodes)
        subset = list(range(subgraph_size))
        signature = build_signature(subset, adj_matrix, node_labels, directed_flag)
        
        # Compute hash (simulate extract_subgraph)
        hash_G = graph_hash(node_labels, [(u, v) for u, v in G.edges()], [0] * len(G.edges()), directed_flag, None)
        
        # Insert into gtrie
        gtrie.insert(signature, hash_G)
    
    return gtrie
