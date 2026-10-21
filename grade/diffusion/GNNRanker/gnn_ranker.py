import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import to_undirected

from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.utils.logger import get_logger


class GINEFrequencyRegressor(nn.Module):
    def __init__(
        self,
        node_classes: int,
        edge_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.node_classes = node_classes
        self.edge_classes = edge_classes
        self.dropout = dropout
        self.node_encoder = nn.Linear(node_classes + 1, hidden_dim)
        self.edge_encoder = nn.Linear(edge_classes, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            update_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(update_mlp, train_eps=True, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _node_features(self, data) -> torch.Tensor:
        if data.x.dim() == 1:
            x = F.one_hot(data.x.long(), num_classes=self.node_classes).float()
        else:
            x = data.x.float()
        if x.shape[-1] != self.node_classes:
            raise ValueError(
                f"Expected {self.node_classes} node features, got {x.shape[-1]}."
            )

        if hasattr(data, "level") and data.level is not None:
            level = data.level.float().reshape(-1, 1)
            graph_max = global_max_pool(level, data.batch).clamp_min(1.0)
            level = level / graph_max[data.batch]
        else:
            level = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        return torch.cat([x, level], dim=-1)

    def _edge_features(self, data) -> torch.Tensor:
        if data.edge_attr is None:
            return torch.ones(
                (data.edge_index.shape[1], self.edge_classes),
                device=data.edge_index.device,
            )
        edge_attr = data.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = F.one_hot(
                edge_attr.long(), num_classes=self.edge_classes
            ).float()
        else:
            edge_attr = edge_attr.float()
        if edge_attr.shape[-1] != self.edge_classes:
            raise ValueError(
                f"Expected {self.edge_classes} edge features, got {edge_attr.shape[-1]}."
            )
        return edge_attr

    def forward(self, data) -> torch.Tensor:
        h = self.node_encoder(self._node_features(data))
        edge_index, edge_attr = to_undirected(
            data.edge_index,
            self._edge_features(data),
            num_nodes=data.x.shape[0],
            reduce="mean",
        )
        edge_attr = self.edge_encoder(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, edge_index, edge_attr)
            update = F.relu(update)
            update = F.dropout(update, p=self.dropout, training=self.training)
            h = norm(h + update)
        graph_embedding = torch.cat(
            [global_add_pool(h, data.batch), global_mean_pool(h, data.batch)], dim=-1
        )
        return self.head(graph_embedding).squeeze(-1)


class GNNRanker(DiffusionAlgorithm):
    """GINE frequency regressor exposed through grade's estimator interface."""

    checkpoint_metric_name = "spearman"
    checkpoint_metric_mode = "max"
    loss_component_names = ("Regression", "Ranking", "Auxiliary")

    def __init__(self, args, dataset_info):
        super().__init__()
        self.device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
        self.rank_loss_weight = args.rank_loss_weight
        self.model = GINEFrequencyRegressor(
            node_classes=dataset_info["n_node_type"],
            edge_classes=dataset_info["n_edge_type"],
            hidden_dim=args.hidden_dim,
            num_layers=args.n_layers,
            dropout=args.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.7, patience=10
            )
            if args.enable_scheduler
            else None
        )

        logger = get_logger()
        logger.info("============= GNN Ranker Settings =============")
        logger.info(f"GPU: {self.device}")
        logger.info(f"Hidden dim: {args.hidden_dim}")
        logger.info(f"GINE layers: {args.n_layers}")
        logger.info(f"Rank-loss weight: {self.rank_loss_weight}")
        logger.info(
            f"# of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    @staticmethod
    def _pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor):
        target_delta = targets[:, None] - targets[None, :]
        score_delta = scores[:, None] - scores[None, :]
        upper_triangle = torch.triu(
            torch.ones_like(target_delta, dtype=torch.bool), diagonal=1
        )
        comparable = upper_triangle & (target_delta != 0)
        if not comparable.any():
            return scores.sum() * 0.0
        direction = torch.sign(target_delta[comparable])
        return F.softplus(-direction * score_delta[comparable]).mean()

    def _losses(self, data):
        data = data.to(self.device)
        scores = self.model(data)
        targets = data.frequency_target.reshape(-1).float()
        regression_loss = F.smooth_l1_loss(scores, targets)
        ranking_loss = self._pairwise_ranking_loss(scores, targets)
        loss = regression_loss + self.rank_loss_weight * ranking_loss
        return loss, regression_loss, ranking_loss, scores, targets

    def train_step(self, data):
        loss, regression_loss, ranking_loss, _, _ = self._losses(data)
        return {
            "loss": loss,
            "loss_X": regression_loss,
            "loss_E": ranking_loss,
            "loss_y": loss.new_zeros(()),
        }

    def validation_step(self, data):
        _, regression_loss, _, scores, targets = self._losses(data)
        return regression_loss, scores, targets

    def test_step(self, data):
        return self.validation_step(data)

    @torch.no_grad()
    def evaluation_epoch(self, data_loader):
        self.model.eval()
        losses = []
        scores = []
        targets = []
        for data in data_loader:
            regression_loss, batch_scores, batch_targets = self.validation_step(data)
            losses.append(regression_loss.item())
            scores.append(batch_scores.cpu())
            targets.append(batch_targets.cpu())
        scores = torch.cat(scores).numpy()
        targets = torch.cat(targets).numpy()
        return {
            "regression_loss": sum(losses) / max(len(losses), 1),
            "spearman": float(spearmanr(scores, targets).statistic),
            "kendall": float(kendalltau(scores, targets).statistic),
        }

    def generate_step(self):
        raise NotImplementedError("GNNRanker is an estimator and cannot generate graphs.")

    @torch.no_grad()
    def estimate_step(self, data):
        data = data.to(self.device)
        score = self.model(data)
        zeros = torch.zeros_like(score)
        return {"G_logp": score, "X_logp": zeros, "E_logp": zeros}
