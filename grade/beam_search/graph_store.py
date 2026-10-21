import torch
import tqdm
import networkx
import igraph
import random
import time
import multiprocessing as mp

from grade.utils.timeout import TimeoutError, timeout

class GraphDatabase():

    def __init__(self, dataset, build_networkx=False):
        self.dataset = dataset
        # self.Gs = [] # for networkx

        self.ig_Gs = [] # for igraph
        self.build_networkx = build_networkx
        if self.build_networkx:
            self.Gs = []
        self.build()

    def build(self):
        for graph_idx in tqdm.tqdm(range(len(self.dataset))):
            datapoint = self.dataset[graph_idx]
            n_node = datapoint.x.shape[0]
            if datapoint.x.dim() == 1:
                x = datapoint.x
            else:
                x = torch.argmax(datapoint.x, dim=-1)
            x = x.cpu().tolist()
            edge_index = datapoint.edge_index.cpu().tolist()
            level = datapoint.level.cpu().tolist()
            left_directed_edges = [edge_idx for edge_idx in range(len(edge_index[0])) if int(level[edge_index[0][edge_idx]]) < int(level[edge_index[1][edge_idx]])]
            edge_list = [(edge_index[0][edge_idx], edge_index[1][edge_idx]) for edge_idx in left_directed_edges]
            edge_attr = torch.argmax(datapoint.edge_attr, dim=-1).cpu().tolist()
            edge_attr = [edge_attr[edge_idx] for edge_idx in left_directed_edges]
            ig_G = igraph.Graph(
                n = n_node,
                directed=True,
                edges=edge_list,
                vertex_attrs = {'operator': x}, 
                edge_attrs = {'edge_type': edge_attr}
            )
            self.ig_Gs.append(ig_G)
        
        if self.build_networkx:
            for graph_idx in tqdm.tqdm(range(len(self.dataset))):
                G = networkx.DiGraph()
                datapoint = self.dataset[graph_idx]
                n_node = datapoint.x.shape[0]
                if datapoint.x.dim() == 1:
                    x = datapoint.x
                else:
                    x = torch.argmax(datapoint.x, dim=-1)
                for node_idx in range(n_node):
                    G.add_node(node_idx, operator=x[node_idx].cpu().tolist())
                edge_index = datapoint.edge_index.cpu().tolist()
                edge_attr = torch.argmax(datapoint.edge_attr, dim=-1).cpu().tolist()
                level = datapoint.level.cpu().tolist()
                for edge_idx in range(len(edge_index[0])):
                    u = edge_index[0][edge_idx]
                    v = edge_index[1][edge_idx]
                    if int(level[u]) < int(level[v]):
                        G.add_edge(u, v, edge_type=edge_attr[edge_idx])
                self.Gs.append(G)
        
    def query(self, graph_idx) -> networkx.DiGraph:
        if self.build_networkx:
            return self.Gs[graph_idx]
        else:
            G = networkx.DiGraph()
            datapoint = self.dataset[graph_idx]
            n_node = datapoint.x.shape[0]
            if datapoint.x.dim() == 1:
                x = datapoint.x
            else:
                x = torch.argmax(datapoint.x, dim=-1)
            for node_idx in range(n_node):
                G.add_node(node_idx, operator=x[node_idx].cpu().tolist())
            edge_index = datapoint.edge_index.cpu().tolist()
            edge_attr = torch.argmax(datapoint.edge_attr, dim=-1).cpu().tolist()
            level = datapoint.level.cpu().tolist()
            for edge_idx in range(len(edge_index[0])):
                u = edge_index[0][edge_idx]
                v = edge_index[1][edge_idx]
                if int(level[u]) < int(level[v]):
                    G.add_edge(u, v, edge_type=edge_attr[edge_idx])
            return G

    def size(self):
        return len(self.ig_Gs)

    def subgraph_match(self, subgraph: networkx.DiGraph):
        sub_ig_G = igraph.Graph.from_networkx(subgraph)
        random_perm = list(range(len(self.ig_Gs)))
        random.shuffle(random_perm)
        result = {
            'graph_idx': -1,
            'S': set(),
            'match': False
        }
        for i, graph_idx in enumerate(random_perm):
            ig_G = self.ig_Gs[graph_idx]
            color1 = [op for op in ig_G.vs['operator']]
            color2 = [op for op in sub_ig_G.vs['operator']]
            
            def cb(g1, g2, mapping_12, mapping_21, *args):
                result['graph_idx'] = graph_idx
                result['S'] = set(mapping_21)
                result['match'] = True
                return False
            
            @timeout(seconds=10)
            def run_vf2_with_timeout(ig_G, sub_ig_G, color1, color2, callback):
                ig_G.subisomorphic_vf2(
                    sub_ig_G,
                    color1 = color1,
                    color2 = color2,
                    callback=callback
                )
            
            try:
                run_vf2_with_timeout(ig_G, sub_ig_G, color1, color2, cb)
            except TimeoutError:
                continue
            
            if result["match"]:
                break
        
        return result

    def subgraph_count(self, subgraph: networkx.DiGraph):
        sub_ig_G = igraph.Graph.from_networkx(subgraph)
        
        total_count = 0
        batch_size = 100
        timeout_per_G = 1
        for graph_idx in tqdm.tqdm(range(len(self.ig_Gs))):
            if graph_idx % batch_size == 0:
                ig_G_batch = []
                color1_batch = []
            
            ig_G = self.ig_Gs[graph_idx]
            color1 = [op for op in ig_G.vs['operator']]
            color2 = [op for op in sub_ig_G.vs['operator']]
            
            ig_G_batch.append(ig_G)
            color1_batch.append(color1)

            if graph_idx % batch_size == batch_size - 1 or graph_idx == len(self.ig_Gs) - 1:
                ctx = mp.get_context('fork')
                counter = ctx.Value('q', 0, lock=False)
                proc = ctx.Process(
                    target=vf2_count_worker_batched,
                    args=(ig_G_batch, sub_ig_G, color1_batch, color2, counter)
                )
                proc.start()
                proc.join(timeout_per_G * len(ig_G_batch))
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                total_count += counter.value
        
        return total_count

    def subgraph_count_mp(self, subgraph: networkx.DiGraph, batch_size=1, max_wait_time=1200, num_workers=None):
        sub_ig_G = igraph.Graph.from_networkx(subgraph)
        color2 = [op for op in sub_ig_G.vs['operator']]
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() // 2)
        manager = mp.Manager()
        left = manager.Value('i', 0)
        left_lock = manager.Lock()
        total = manager.Value('q', 0)
        total_lock = manager.Lock()
        ctx = mp.get_context('fork')
    
        def _worker(counter):
            while True:
                with left_lock:
                    start = left.value
                    left.value += batch_size
                if start >= len(self.ig_Gs):
                    break
                end = min(start + batch_size, len(self.ig_Gs))
                ig_G_batch = []
                color1_batch = []
                for idx in range(start, end):
                    ig_G = self.ig_Gs[idx]
                    color1 = [op for op in ig_G.vs['operator']]
                    ig_G_batch.append(ig_G)
                    color1_batch.append(color1)
                vf2_count_worker_batched(ig_G_batch, sub_ig_G, color1_batch, color2, counter)

        procs = []
        counters = []
        n_batches = (len(self.ig_Gs) + batch_size - 1) // batch_size
        for _ in range(max(1, min(num_workers, n_batches))):
            counter = ctx.Value('q', 0, lock=True)
            p = ctx.Process(target=_worker, args=(counter,))
            p.start()
            procs.append(p)
            counters.append(counter)
        start_time = time.time()
        for i, p in enumerate(procs):
            remaining_time = max(0, max_wait_time - (time.time() - start_time))
            p.join(remaining_time)
            if p.is_alive():
                with total_lock:
                    with counters[i].get_lock():
                        total.value += counters[i].value
                p.terminate()
            else:
                with total_lock:
                    with counters[i].get_lock():
                        total.value += counters[i].value
        return int(total.value)

    def subgraph_count_with_timeout(self, subgraph: networkx.DiGraph):        
        sub_ig_G = igraph.Graph.from_networkx(subgraph)

        total_count = 0
        def cb(g1, g2, mapping_12, mapping_21, *args):
            nonlocal total_count
            total_count += 1
            return True

        @timeout(seconds=600)
        def run_vf2_with_timeout(sub_ig_G, callback):
            for i, graph_idx in enumerate(tqdm.tqdm(range(len(self.ig_Gs)))):
                ig_G = self.ig_Gs[graph_idx]
                color1 = [op for op in ig_G.vs['operator']]
                color2 = [op for op in sub_ig_G.vs['operator']]        
            
                ig_G.subisomorphic_vf2(
                    sub_ig_G,
                    color1 = color1,
                    color2 = color2,
                    callback=callback
                )
        
        try:
            run_vf2_with_timeout(sub_ig_G, cb)
        except:
            pass

        return total_count

    def automorphism_count(self, subgraph):
        sub_ig_G = igraph.Graph.from_networkx(subgraph)
        color = [int(x) for x in sub_ig_G.vs['operator']]
        ecolor = [int(t) for t in sub_ig_G.es['edge_type']]
        
        return sub_ig_G.count_isomorphisms_vf2(
            sub_ig_G,
            color1=color, color2=color,
            edge_color1=ecolor, edge_color2=ecolor
        )

def vf2_count_worker_batched(ig_G_list, sub_ig_G, color1_list, color2, counter):
    local_count = 0
    flush_interval = 1000000
    for ig_G, color1 in zip(ig_G_list, color1_list):
        def cb(g1, g2, mapping_12, mapping_21, *args):
            nonlocal local_count
            nonlocal counter
            local_count += 1
            if local_count >= flush_interval:
                with counter.get_lock():
                    counter.value += local_count
                local_count = 0
            return True
        ig_G.subisomorphic_vf2(sub_ig_G, color1=color1, color2=color2, callback=cb)
        with counter.get_lock():
            counter.value += local_count