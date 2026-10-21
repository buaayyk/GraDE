from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_dense_batch

from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.utils.logger import get_logger


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()         # GCNConv
        self.norms = nn.ModuleList()         # LayerNorm
        # Dimension list for each layer
        dims = [in_dim] + [hidden_dim] * num_layers
        # Build GCNConv+LayerNorm pairs
        for in_ch, out_ch in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(in_ch, out_ch))   # [GCNConv(), GCNConv(), ...]
            self.norms.append(nn.LayerNorm(out_ch))
        self.dropout = dropout                          # to forward
        # Predict node-wise Gaussian parameters from hidden_dim features
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    # x[num_nodes, in_dim], edge_index[2, num_edges]
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Map node hidden states to mu/logvar
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar


class NodeDecoder(nn.Module):
    # latent -> hidden -> hidden -> out
    # out_dim = node_classes
    # Node latent -> node class logits
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class EdgeDecoder(nn.Module):
    # Similar to NodeDecoder but for edge pairs (i, j)
    # Input dim: 2 * latent_dim, output dim = edge_classes
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    # z shape [B, N, D], build pair features -> [B, N, N, edge_classes]
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Expand to get zi, zj with shape [B, N, N, D]
        zi = z.unsqueeze(2).expand(-1, -1, z.size(1), -1)
        zj = z.unsqueeze(1).expand(-1, z.size(1), -1, -1)
        # Collect pair features [B, N, N, 2D]
        pair_features = torch.cat([zi, zj], dim=-1)
        # Predict edge class for every node pair
        return self.mlp(pair_features)


class GraphVAENet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        decoder_hidden_dim: int,
        num_layers: int,
        dropout: float,
        node_classes: int,
        edge_classes: int,
    ):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.node_decoder = NodeDecoder(latent_dim, decoder_hidden_dim, node_classes, dropout)
        self.edge_decoder = EdgeDecoder(latent_dim, decoder_hidden_dim, edge_classes, dropout)

    def decode(self, z_dense: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_logits = self.node_decoder(z_dense)
        edge_logits = self.edge_decoder(z_dense)
        return node_logits, edge_logits


# E shape: [B, N, N, edge_classes]
def encode_no_edge(E: torch.Tensor) -> torch.Tensor:
    if E.size(-1) == 0:
        return E
    # Detect entries without any edge feature
    no_edge = torch.sum(E, dim=-1) == 0
    first_channel = E[..., 0]
    first_channel[no_edge] = 1.0
    E[..., 0] = first_channel
    # Remove self-loop channels
    diag = torch.eye(E.size(1), dtype=torch.bool, device=E.device).unsqueeze(0).unsqueeze(-1)
    E = E.masked_fill(diag, 0.0)
    return E


def batch_to_dense(data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Pad nodes into dense tensors
    X_dense, node_mask = to_dense_batch(data.x, data.batch)
    # Remove self-loops to avoid leaking them to training
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    max_nodes = X_dense.size(1)
    # Some datasets provide no edge attributes; add dummy ones
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), 1, device=data.x.device)

    # Convert sparse edges to dense adjacency tensor
    E_dense = to_dense_adj(edge_index, batch=data.batch, edge_attr=edge_attr, max_num_nodes=max_nodes)
    E_dense = encode_no_edge(E_dense)
    return X_dense, E_dense, node_mask


class GraphVAE(DiffusionAlgorithm):
    def __init__(self, args, train_set, val_set, test_set):
        super().__init__()
        if torch.cuda.is_available():
            device_arg = str(args.device)
            self.device = device_arg if device_arg.startswith('cuda') else f'cuda:{device_arg}'
        else:
            self.device = 'cpu'
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.encoder_layers = args.encoder_layers
        self.dropout = args.dropout
        self.kl_beta = args.kl_beta
        self.n_sample = args.n_sample
        self.rounds = args.rounds

        example_data = train_set[0]
        in_dim = example_data.x.size(-1)

        self.dataset_info = train_set.get_dataset_info()
        self.max_nodes = self.dataset_info["max_n_nodes"]
        # one-hot
        self.node_classes = self.dataset_info["n_node_type"]
        self.edge_classes = self.dataset_info["n_edge_type"]
        # Used for decoder output dim, CE loss, and generation one-hot

        # Sample graph size from empirical distribution
        node_distribution = torch.tensor(self.dataset_info["n_node_distribution"]).float()
        # Disallow zero-node samples
        node_distribution[0] = 0
        if node_distribution.sum() == 0:
            node_distribution = torch.ones_like(node_distribution)
        self.node_distribution = (node_distribution / node_distribution.sum()).to(self.device)
        self.node_count_ids = torch.arange(len(node_distribution), device=self.device)

        self.logger = get_logger()
        self.logger.info("============= GraphVAE Settings =============")
        self.logger.info(f"GPU: {self.device}")
        self.logger.info(f"Dataset: {args.dataset}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Latent dim: {self.latent_dim}")
        self.logger.info(f"Hidden dim: {self.hidden_dim}")
        self.logger.info(f"Decoder hidden dim: {self.decoder_hidden_dim}")
        self.logger.info(f"Encoder layers: {self.encoder_layers}")
        self.logger.info(f"KL beta: {self.kl_beta}")

        self.model = GraphVAENet(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            num_layers=self.encoder_layers,
            dropout=self.dropout,
            node_classes=self.node_classes,
            edge_classes=self.edge_classes,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, verbose=False
        ) if args.enable_scheduler else None


    @overrides
    def get_model(self):
        return self.model

    @overrides
    def get_optimizer(self):
        return self.optimizer

    @overrides
    def get_scheduler(self):
        return self.scheduler


    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _forward_internal(self, data: Data, training: bool):
        data = data.to(self.device)
        # Convert sparse graph to dense tensors
        dense_X, dense_E, node_mask = batch_to_dense(data)
        # Encode node features
        mu, logvar = self.model.encoder(data.x, data.edge_index)
        # Sample node latent vectors
        z = self._reparameterize(mu, logvar, training=training)
        # Convert z to dense format
        dense_z, _ = to_dense_batch(z, data.batch, max_num_nodes=dense_X.size(1))
        # Decode node/edge logits
        node_logits, edge_logits = self.model.decode(dense_z)

        # Mask self-loop logits
        diag_mask = torch.eye(dense_z.size(1), dtype=torch.bool, device=dense_z.device).unsqueeze(0).unsqueeze(-1)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)

        return node_logits, edge_logits, mu, logvar, dense_X, dense_E, node_mask

    def _compute_losses(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        dense_X: torch.Tensor,
        dense_E: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        node_targets = torch.argmax(dense_X, dim=-1)
        valid_nodes = node_mask    # True for valid nodes, False for padded nodes
        if valid_nodes.sum() == 0:
            loss_X = torch.tensor(0.0, device=node_logits.device)
        else:
            loss_X = F.cross_entropy(node_logits[valid_nodes], node_targets[valid_nodes])

        # dense_E: [B, N, N, edge_classes] -> [B, N, N]
        edge_targets = torch.argmax(dense_E, dim=-1)
        # [B, N] -> [B, 1, N] & [B, N, 1] -> [B, N, N]
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        diag = torch.eye(edge_mask.size(1), dtype=torch.bool, device=edge_mask.device).unsqueeze(0)
        edge_mask = edge_mask & (~diag)
        if edge_mask.sum() == 0:
            loss_E = torch.tensor(0.0, device=node_logits.device)
        else:
            loss_E = F.cross_entropy(edge_logits[edge_mask], edge_targets[edge_mask])

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl = kl.mean()

        total_loss = loss_X + loss_E + self.kl_beta * kl

        return total_loss, loss_X, loss_E, kl

    @overrides
    def train_step(self, data):
        node_logits, edge_logits, mu, logvar, dense_X, dense_E, node_mask = self._forward_internal(data, training=True)
        total_loss, loss_X, loss_E, kl = self._compute_losses(node_logits, edge_logits, mu, logvar, dense_X, dense_E, node_mask)
        return {"loss": total_loss, "loss_X": loss_X, "loss_E": loss_E, "loss_y": kl * 0}  # Unsupervised, keep label loss zero

    @overrides
    def validation_step(self, data):
        with torch.no_grad():
            node_logits, edge_logits, mu, logvar, dense_X, dense_E, node_mask = self._forward_internal(
                data, training=False
            )
            total_loss, _, _, _ = self._compute_losses(node_logits, edge_logits, mu, logvar, dense_X, dense_E, node_mask)
        return {"loss": total_loss}


    @overrides
    def test_step(self, data):
        with torch.no_grad():
            node_logits, edge_logits, _, _, dense_X, dense_E, node_mask = self._forward_internal(data, training=False)

        X_idx = torch.argmax(dense_X, dim=-1).long()  # -> [B, N]
        E_idx = torch.argmax(dense_E, dim=-1).long()  # -> [B, N, N]

        # Assign padding nodes to special id
        X_idx[~node_mask] = self.node_classes
        edge_pad_mask = ~(node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
        diag = torch.eye(node_mask.size(1), dtype=torch.bool, device=node_mask.device).unsqueeze(0)
        edge_pad_mask = edge_pad_mask | diag
        E_idx[edge_pad_mask] = self.edge_classes

        node_logits = node_logits.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        edge_logits = edge_logits.masked_fill(edge_pad_mask.unsqueeze(-1), 0.0)

        return node_logits, edge_logits, X_idx, E_idx


    @overrides
    def generate_step(self):
        self.model.eval()
        generated_graphs = []
        for _ in range(self.n_sample):
            num_nodes = self._sample_num_nodes()
            z = torch.randn(1, num_nodes, self.latent_dim, device=self.device)   # shape
            node_logits, edge_logits = self.model.decode(z)
            node_probs = F.softmax(node_logits, dim=-1)[0]
            edge_probs = F.softmax(edge_logits, dim=-1)[0]

            node_samples = torch.multinomial(node_probs, 1).squeeze(-1)
            x = F.one_hot(node_samples, num_classes=self.node_classes).float()

            edge_samples = torch.multinomial(edge_probs.reshape(-1, self.edge_classes), 1).view(num_nodes, num_nodes)
            edge_index = []
            edge_attr = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                    edge_type = edge_samples[i, j].item()
                    if edge_type == 0:
                        continue
                    edge_index.append([i, j])
                    edge_attr.append(edge_type)

            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
                edge_attr = F.one_hot(
                    torch.tensor(edge_attr, dtype=torch.long, device=self.device),
                    num_classes=self.edge_classes,
                ).float()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.empty((0, self.edge_classes), dtype=torch.float, device=self.device)
            generated_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return generated_graphs

    def _sample_num_nodes(self) -> int:
        sampled_idx = torch.multinomial(self.node_distribution, 1).item()
        sampled_idx = max(sampled_idx, 1)
        return int(sampled_idx)

    @overrides
    def estimate_step(self, data):
        with torch.no_grad():
            node_logits, edge_logits, mu, logvar, dense_X, dense_E, node_mask = self._forward_internal(data, training=False)
            # Resample z for stochastic estimate
            z_sample = self._reparameterize(mu, logvar, training=True)
            dense_z_sample, _ = to_dense_batch(z_sample, data.batch, max_num_nodes=dense_X.size(1))
            node_logits, edge_logits = self.model.decode(dense_z_sample)
            diag_mask = torch.eye(dense_z_sample.size(1), dtype=torch.bool, device=dense_z_sample.device).unsqueeze(0).unsqueeze(-1)
            edge_logits = edge_logits.masked_fill(diag_mask, -1e9)

            # logits -> log p
            node_log_probs = F.log_softmax(node_logits, dim=-1)
            edge_log_probs = F.log_softmax(edge_logits, dim=-1)

            # node: [B, N, node_classes] -> [B, N]
            # edge: [B, N, N, edge_classes] -> [B, N, N]
            node_targets = torch.argmax(dense_X, dim=-1)
            edge_targets = torch.argmax(dense_E, dim=-1)

            # build mask, [B, N], [B, N, N]
            node_mask_float = node_mask.float()
            edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).float()
            diag = torch.eye(edge_mask.size(1), dtype=edge_mask.dtype, device=edge_mask.device).unsqueeze(0)
            edge_mask = edge_mask * (1 - diag)

            # [B, N, node_classes] -> [B, N, 1] -> [B, N] (gather)
            gather_X = torch.gather(node_log_probs, -1, node_targets.unsqueeze(-1)).squeeze(-1)
            gather_E = torch.gather(edge_log_probs, -1, edge_targets.unsqueeze(-1)).squeeze(-1)

            # Sum the log p of whole graph
            X_logp = (gather_X * node_mask_float).sum(dim=1)
            E_logp = (gather_E * edge_mask).sum(dim=(1, 2))

            node_batch = data.batch.to(mu.device)
            kl_per_node = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            num_graphs = node_mask.size(0)
            kl = torch.zeros(num_graphs, device=mu.device).scatter_add_(0, node_batch, kl_per_node)
            G_logp = X_logp + E_logp - self.kl_beta * kl

        return {
            "G_logp": G_logp,
            "X_logp": X_logp,
            "E_logp": E_logp,
        }
