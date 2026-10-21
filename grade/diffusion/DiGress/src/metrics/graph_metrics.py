import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import torch.nn as nn
from src.metrics.molecular_metrics import GeneratedNDistribution, GeneratedNodesDistribution, GeneratedEdgesDistribution,HistogramsMAE
# from src.analysis.rdkit_functions import compute_molecular_metrics

class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)
        
        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]
        
        target = target[:,self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


def create_ce_class(class_name):
    return type(class_name, (CEPerClass,), {
        '__init__': lambda self, i: CEPerClass.__init__(self, i)
    })


class EdgeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)
        
class NoEdgeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NodeMetrics(MetricCollection):
    def __init__(self, dataset_infos):

        class_dict = {}
        for op_name, op_id in dataset_infos['encoder'].items():
            class_dict[op_name] = create_ce_class(f"{op_name}_CE")
        
        metrics_list = []
        for op_id, op_name in enumerate(dataset_infos['decoder']):
            metrics_list.append(class_dict[op_name](op_id))
        
        super().__init__(metrics_list)
        
        
class EdgeMetrics(MetricCollection):
    def __init__(self):
        edge1 = EdgeCE(0)
        #edge2=NoEdgeCE(1)
        super().__init__([edge1])

        
class TrainMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_node_metrics = NodeMetrics(dataset_infos=dataset_infos)
        self.train_edge_metrics = EdgeMetrics()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E):
        self.train_node_metrics(masked_pred_X, true_X)
        self.train_edge_metrics(masked_pred_E, true_E)
        
    def reset(self):
        for metric in [self.train_node_metrics, self.train_edge_metrics]:
            metric.reset()


class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di['max_n_nodes'])
        self.generated_node_dist = GeneratedNodesDistribution(di['n_node_type'])
        self.generated_edge_dist = GeneratedEdgesDistribution(di['n_edge_type'])
        
        n_target_dist = torch.tensor(di["n_node_distribution"]).type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer('n_target_dist', n_target_dist)

        node_target_dist = torch.tensor(di["X_marginal"]).type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer('node_target_dist', node_target_dist)

        edge_target_dist = torch.tensor(di["E_marginal"]).type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer('edge_target_dist', edge_target_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)

        self.dataset_info = di

    def forward(self, molecules: list):

        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.compute()
        self.edge_dist_mae(generated_edge_dist)

    def reset(self):
        for metric in [self.n_dist_mae, self.node_dist_mae, self.edge_dist_mae]:
            metric.reset()
