import torch
import torch_geometric
import networkx
import torch.nn.functional as F
import numpy as np

def tensor_to_networkx(datapoint: torch_geometric.data.Data, directed_flag=True):
    X = (torch.argmax(datapoint.x, dim=-1) if datapoint.x.dim() == 2 else datapoint.x).cpu().tolist()
    edge_index = datapoint.edge_index.cpu().tolist()
    edge_attr = torch.argmax(datapoint.edge_attr, dim=-1).cpu().tolist()
    level = datapoint.level.cpu().tolist() if directed_flag else None
    G = networkx.DiGraph()
    for i in range(len(X)):
        G.add_node(i, operator=X[i])
    for edge_idx in range(len(edge_index[0])):
        u = edge_index[0][edge_idx]
        v = edge_index[1][edge_idx]
        if not directed_flag or level[u] < level[v]:
            G.add_edge(u, v, edge_type=edge_attr[edge_idx])
    return G

def networkx_to_tensor(G: networkx.DiGraph, n_node_type, n_edge_type):
    directed_flag = False
    for u, v in G.edges():
        if (v, u) not in G.edges():
            directed_flag = True
            break
    node_mapping = {}
    ops = []
    count = 0
    for node_idx in G.nodes():
        node_mapping[node_idx] = count
        count += 1
        ops.append(G.nodes[node_idx]['operator'])
    ops = torch.tensor(ops)
    x = F.one_hot(ops, num_classes=n_node_type).float()
    edge_index = [[],[]]
    edge_attr = []
    for u, v in G.edges():
        edge_index[0].append(node_mapping[u])
        edge_index[1].append(node_mapping[v])
        edge_attr.append(G.edges[u, v]['edge_type'])
    edge_index = torch.tensor(edge_index).long()
    edge_attr = torch.tensor(edge_attr)
    edge_attr = F.one_hot(edge_attr, num_classes=n_edge_type) if edge_attr.dim() == 1 else edge_attr
    y = torch.tensor([[0]]).float()
    if directed_flag:
        level = np.zeros((G.number_of_nodes()))
        for v in networkx.topological_sort(G):
            for u in G.predecessors(v):
                level[node_mapping[v]] = max(level[node_mapping[v]], level[node_mapping[u]] + 1)
        level = torch.tensor(level).float()
        datapoint = torch_geometric.data.Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y, level=level)
    else:
        datapoint = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return datapoint