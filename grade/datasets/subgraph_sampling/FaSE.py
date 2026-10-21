import random
from dataclasses import dataclass
import pynauty
import multiprocessing as mp

from grade.datasets.subgraph_sampling.GTRIE import (
    NO_EDGE,
    _discrete_edge_types,
    _discrete_node_labels,
)
from grade.utils.timeout import timeout, TimeoutError

import multiprocessing as mp


def compute_canonical_signature_from_adj(adj_map, node_labels, subset, directed_flag):
    if not subset:
        return ""
    
    idx_map = {orig: i for i, orig in enumerate(subset)}
    base_n = len(subset)
    adj = {}
    total_vertices = base_n
    edge_nodes = []
    
    for i in range(base_n):
        adj[i] = set()
    
    for orig_u in subset:
        for orig_v, edge_type in adj_map[orig_u].items():
            if orig_v not in idx_map:
                continue
            u = idx_map[orig_u]
            v = idx_map[orig_v]
            if not directed_flag and u > v:
                continue  
            edge_node_id = total_vertices
            total_vertices += 1
            edge_nodes.append((edge_node_id, edge_type))
            adj.setdefault(edge_node_id, set())
            
            adj[u].add(edge_node_id)
            adj[edge_node_id].add(v)
            if not directed_flag:
                adj[v].add(edge_node_id)
                adj[edge_node_id].add(u)
    g = pynauty.Graph(number_of_vertices=total_vertices, directed=directed_flag)
    for v in range(total_vertices):
        g.connect_vertex(v, list(adj.get(v, [])))
    
    color_classes = {}
    for orig, new_idx in idx_map.items():
        lab = int(node_labels[orig])
        color_classes.setdefault(("node", lab), []).append(new_idx)
    for edge_node_id, edge_type in edge_nodes:
        color_classes.setdefault(("edge", int(edge_type)), []).append(edge_node_id)
    coloring = [set(cls) for cls in color_classes.values()]
    g.set_vertex_coloring(coloring)
    return pynauty.certificate(g)

def _canonical_signature_worker(adj_map, node_labels, subset, directed_flag, queue):
    result = compute_canonical_signature_from_adj(adj_map, node_labels, subset, directed_flag)
    queue.put(result)
    
def compute_canonical_signature_with_timeout(adj_map, node_labels, subset, directed_flag, timeout=10):
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    p = ctx.Process(
        target=_canonical_signature_worker, 
        args=(adj_map, node_labels, subset, directed_flag, queue)
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        
    if queue.empty():
        return None
    
    result = queue.get()
    return result

@dataclass
class FaSEOccurrence:
    graph_idx: int
    nodes: tuple
    sample_prob: float
    sample_weight: float

class NeighbourCache:
    def __init__(self, data, directed_flag):
        self.data = data
        self.directed_flag = directed_flag

    def get(self, idx):
        return build_sparse_representation(self.data[idx], self.directed_flag)


def build_neighbour_cache(data, directed_flag):
    return NeighbourCache(data, directed_flag)


def build_sparse_representation(datapoint, directed_flag):
    n_node = datapoint.x.shape[0]
    adj_map = [dict() for _ in range(n_node)]  
    neighbors = [set() for _ in range(n_node)]

    node_labels = _discrete_node_labels(datapoint.x)
    if hasattr(node_labels, "tolist"):
        node_labels = node_labels.tolist()
    node_labels = [int(x) for x in node_labels]

    edge_index = datapoint.edge_index.cpu().tolist()
    edge_types = _discrete_edge_types(datapoint.edge_attr)
    if hasattr(edge_types, "tolist"):
        edge_types = edge_types.tolist()

    if directed_flag:
        level = datapoint.level.cpu().tolist()
    else:
        level = None

    for edge_idx in range(len(edge_index[0])):
        u = int(edge_index[0][edge_idx])
        v = int(edge_index[1][edge_idx])
        attr = int(edge_types[edge_idx]) if edge_types is not None else 0
        if directed_flag:
            if int(level[u]) < int(level[v]):
                adj_map[u][v] = attr
                neighbors[u].add(v)
                neighbors[v].add(u)
        else:
            adj_map[u][v] = attr
            adj_map[v][u] = attr
            neighbors[u].add(v)
            neighbors[v].add(u)
    return adj_map, neighbors, node_labels

class FaSETrieNode:
    def __init__(self, signature=()):
        self.children = {}     
        self.count = 0.0        
        self.signature = signature  
        self.instances = []     
        self.canonical_signature = None  

class FaSEEngine:
    def __init__(self, subgraph_size, tau, directed_flag):
        self.subgraph_size = subgraph_size
        self.directed_flag = directed_flag
        self.p_list = self._build_probabilities(subgraph_size, tau)
        self.root = FaSETrieNode(signature=tuple())
        self.total_weight = 0.0     # leaf

    def _build_probabilities(self, k, tau):
        if k <= 1:
            return []
        probs = []
        for depth in range(k - 1):
            if depth == 0 or depth == k - 2:
                probs.append(1.0)  # p_0 = p_{K-1} = 1
            else:
                i = depth + 2  
                probs.append((1 - i / (k + 1)) ** tau)
        return probs

    def process_graph(self, graph_idx, adj_map, neighbors, node_labels, rng, max_samples_per_graph=None):
        n = len(neighbors)
        roots = list(range(n))
        rng.shuffle(roots)
        samples_collected = [0]
        for root in roots:
            if max_samples_per_graph is not None and samples_collected[0] >= max_samples_per_graph:
                break
            current = [root]   
            root_label = self._ls_label(adj_map, node_labels, current)
            child = self.root.children.get(root_label)
            if child is None:
                child = FaSETrieNode(signature=self.root.signature + (root_label,))
                self.root.children[root_label] = child
            extension = {v for v in neighbors[root] if v > root}   
            should_continue = self._dfs(        
                graph_idx,
                adj_map,
                neighbors,
                node_labels,
                current,
                extension,
                root,
                child,
                depth=0,        
                path_prob=1.0,  
                rng=rng,
                samples_collected=samples_collected,
                max_samples_per_graph=max_samples_per_graph,
            )
            if not should_continue:
                break
    def _dfs(self, graph_idx, adj_map, neighbors, node_labels, current, extension, root, trie_node, depth, path_prob, rng, samples_collected, max_samples_per_graph):
        if max_samples_per_graph is not None and samples_collected[0] >= max_samples_per_graph:
            return False
        if len(current) == self.subgraph_size:
            subset_sorted = tuple(sorted(current))
            if trie_node.canonical_signature is None:
                if self.subgraph_size <= 6:
                    res = self._canonical_signature_from_adj(adj_map, node_labels, subset_sorted)
                else:
                    res = compute_canonical_signature_with_timeout(adj_map, node_labels, subset_sorted, self.directed_flag, 10)
                if res is None:
                    return False
                else:
                    trie_node.canonical_signature = res

                # try:
                #     trie_node.canonical_signature = compute_canonical_signature_with_timeout(
                #         adj_map, node_labels, subset_sorted, self.directed_flag
                #     )
                # except (TimeoutError, Exception):
                #     # Skip this instance on timeout or error
                #     return True
                
            weight = 1.0 / max(path_prob, 1e-12)
            self.total_weight += weight     
            trie_node.count += weight       
            
            occurrence = FaSEOccurrence(
                graph_idx=graph_idx,
                nodes=subset_sorted,
                sample_prob=path_prob,
                sample_weight=weight,
            )
            trie_node.instances.append(occurrence)
            samples_collected[0] += 1
            if max_samples_per_graph is not None and samples_collected[0] >= max_samples_per_graph:
                return False
            return True

        
        extension_list = sorted(extension)
        base_extension = set(extension_list)
        for idx, w in enumerate(extension_list):
            cont_prob = self.p_list[depth] if depth < len(self.p_list) else 1.0
            if cont_prob < 1.0 and rng.random() > cont_prob:
                continue
            new_prob = path_prob * cont_prob
            new_current = current + [w]
            # LS-label
            label = self._ls_label(adj_map, node_labels, new_current)   
            child = trie_node.children.get(label)                         
            if child is None:
                child = FaSETrieNode(signature=trie_node.signature + (label,))
                trie_node.children[label] = child

            new_extension = set(extension_list[idx + 1 :])   
            for neighbor in neighbors[w]:                    
                if neighbor > root and neighbor not in new_current and neighbor not in base_extension:
                    new_extension.add(neighbor)
            
            should_continue = self._dfs(
                graph_idx,
                adj_map,
                neighbors,
                node_labels,
                new_current,
                new_extension,
                root,
                child,
                depth + 1,
                new_prob,
                rng,
                samples_collected,
                max_samples_per_graph,
            )
            if not should_continue:
                return False
        return True

    def _ls_label(self, adj_map, node_labels, subset):
        last = subset[-1]               
        prev_nodes = sorted(subset[:-1])  
        label_head = (int(node_labels[last]),)
        edges = []
        for prev in prev_nodes:
            edges.append(self._edge_type(adj_map, last, prev))  
            if self.directed_flag:
                edges.append(self._edge_type(adj_map, prev, last))
        return label_head + tuple(edges)

    def collect_results(self):
        freq = {}
        self._collect(self.root, freq)
        return freq

    def _collect(self, node, freq):
        if not node.children and node.instances:
            key = node.canonical_signature if node.canonical_signature is not None else tuple()
            record = freq.get(key)
            if record is None:
                record = {"estimated_count": 0.0, "occurrences": []}
                freq[key] = record
            record["estimated_count"] += node.count
            record["occurrences"].extend(node.instances)
            return
        for child in node.children.values():    
            self._collect(child, freq)

    def _canonical_signature_from_adj(self, adj_map, node_labels, subset):
        return compute_canonical_signature_from_adj(adj_map, node_labels, subset, self.directed_flag)
        # if not subset:
        #     return ""
        
        # idx_map = {orig: i for i, orig in enumerate(subset)}
        # base_n = len(subset)
        # adj = {}
        # total_vertices = base_n
        # edge_nodes = []
        
        # for i in range(base_n):
        #     adj[i] = set()
        
        # for orig_u in subset:
        #     for orig_v, edge_type in adj_map[orig_u].items():
        #         if orig_v not in idx_map:
        #             continue
        #         u = idx_map[orig_u]
        #         v = idx_map[orig_v]
        #         if not self.directed_flag and u > v:
        #             continue  
        #         edge_node_id = total_vertices
        #         total_vertices += 1
        #         edge_nodes.append((edge_node_id, edge_type))
        #         adj.setdefault(edge_node_id, set())
                
        #         adj[u].add(edge_node_id)
        #         adj[edge_node_id].add(v)
        #         if not self.directed_flag:
        #             adj[v].add(edge_node_id)
        #             adj[edge_node_id].add(u)
        # g = pynauty.Graph(number_of_vertices=total_vertices, directed=self.directed_flag)
        # for v in range(total_vertices):
        #     g.connect_vertex(v, list(adj.get(v, [])))
        
        # color_classes = {}
        # for orig, new_idx in idx_map.items():
        #     lab = int(node_labels[orig])
        #     color_classes.setdefault(("node", lab), []).append(new_idx)
        # for edge_node_id, edge_type in edge_nodes:
        #     color_classes.setdefault(("edge", int(edge_type)), []).append(edge_node_id)
        # coloring = [set(cls) for cls in color_classes.values()]
        # g.set_vertex_coloring(coloring)
        # return pynauty.certificate(g)

    def _edge_type(self, adj_map, u, v):
        return adj_map[u].get(v, NO_EDGE)
