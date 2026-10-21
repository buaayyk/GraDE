import torch
from losses import CELoss
from overrides import overrides

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.utils import index_to_mask

from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.utils.logger import get_logger
from dataset_info import DatasetInfo, get_dataset_info, get_train_smiles
from forward_diff import ForwardDiffusion
from digress_models import GraphTransformer
from models import MPNN
from losses import CELoss
from sampling import TauLeaping
from eval_molecule import *
from auxiliary_features import AuxFeatures, ExtraMolecularFeatures
from utils import to_dense, add_mask_idx
from utils import *
from rdkit_functions import BasicMolecularMetrics, graph2smile

# ---------- For estimator ----------
from estimating import Estimator
# ---------- For estimator ----------

class DisCo(DiffusionAlgorithm):

    def __init__(self, args, train_set, val_set, test_set):
        super().__init__()
        # Store parser args
        """ ========================== General training settings ========================== """
        self.device = args.device
        self.batch_size = args.batch_size
        self.graph_domain = args.graph_domain
        self.backbone = args.backbone
        self.BAR = args.BAR
        """ ========================== Diffusion settings ========================== """
        self.diff_type = args.diff_type
        self.min_time = args.min_time
        self.sampling_steps = args.sampling_steps
        self.n_sample = args.n_sample
        self.beta = args.beta
        self.alpha = args.alpha
        """ ========================== Parametric model settings ========================== """
        self.lr = args.lr
        self.wd = args.wd
        self.enable_scheduler = args.enable_scheduler
        self.dropout = args.dropout
        self.lambda_train = args.lambda_train
        self.n_layers = args.n_layers
        self.n_dim = args.n_dim
        self.cycle_fea = args.cycle_fea
        self.eigen_fea = args.eigen_fea
        self.rwpe_fea = args.rwpe_fea
        self.global_fea = args.global_fea
        self.level_fea = args.level_fea
        self.extra_time = args.extra_time

        self.device = 'cpu' if self.device == 'cpu' else 'cuda:' + self.device
        self.BAR = True if self.BAR == 1 else False
        self.cycle_fea = True if self.cycle_fea == 1 else False
        self.eigen_fea = True if self.eigen_fea == 1 else False
        self.rwpe_fea = False
        self.global_fea = True if self.global_fea == 1 else False
        self.level_fea = self.level_fea
        self.aux_feas = [self.cycle_fea, self.eigen_fea, self.rwpe_fea, self.global_fea]
        self.extra_time = True if self.extra_time == 1 else False

        self.n_node_type = val_set[0].x.shape[-1]
        self.n_edge_type = val_set[0].edge_attr.shape[-1] # including the absense of an edge
        self.dataset_info = val_set.get_dataset_info()
        self.max_n_nodes, self.n_node_distribution, self.E_marginal, self.X_marginal = self.dataset_info["max_n_nodes"], self.dataset_info["n_node_distribution"], self.dataset_info["E_marginal"], self.dataset_info["X_marginal"]
        assert len(self.E_marginal) == self.n_edge_type
        assert len(self.X_marginal) == self.n_node_type
        self.dataset_info["n_edge_type"], self.dataset_info["n_node_type"] = self.n_edge_type, self.n_node_type
        for key, value in self.dataset_info.items():
            setattr(DatasetInfo, key, value)
        self.dataset_info = DatasetInfo
        self.n_node_distribution = torch.distributions.categorical.Categorical(torch.tensor(self.n_node_distribution))
        self.E_marginal = torch.tensor(self.E_marginal).float().to(self.device)
        self.X_marginal = torch.tensor(self.X_marginal).float().to(self.device)
        self.logger = get_logger()

        self.logger.info("============= DisCo General Settings =============")
        self.logger.info('GPU: {}'.format(self.device))
        self.logger.info("Dataset: {}".format(args.dataset)) # TODO: DisCo shouldn't konw args.dataset
        self.logger.info("Batch size: {}".format(self.batch_size))
        self.logger.info("Backbone: {}".format(self.backbone))
        self.logger.info("lr: {}".format(self.lr))
        self.logger.info("wd: {}".format(self.wd))
        self.logger.info("dropout: {}".format(self.dropout))
        self.logger.info("# of layers: {}".format(self.n_layers))
        self.logger.info("Emb Dim: {}".format(self.n_dim))
        self.logger.info("============= DisCo Diffusion Settings =============")
        self.logger.info("Diffusion Type: {}".format(self.diff_type))
        self.logger.info("Time exponential: {}".format(self.beta))
        self.logger.info("Time base: {}".format(self.alpha))
        self.logger.info("# of samples: {}".format(self.n_sample))
        self.logger.info("Sampling steps: {}".format(self.sampling_steps))
        self.logger.info('E marginal: {}'.format(self.E_marginal.tolist()))
        self.logger.info('X marginal: {}'.format(self.X_marginal.tolist()))
        self.logger.info(f'Lambda Train: {self.lambda_train}')
        self.logger.info(f'Extra Time: {self.extra_time}')
        self.logger.info("")

        """ ========================== Losses ========================== """
        self.train_loss = CELoss
        """ ========================== Models ========================== """        
        if self.graph_domain == 'mol':
            self.extra_molecule_feature = ExtraMolecularFeatures(self.dataset_info)
        else:
            self.extra_molecule_feature = None
        self.add_auxiliary_feature = AuxFeatures(self.aux_feas, self.max_n_nodes, self.extra_molecule_feature)
        self.diffuser = ForwardDiffusion(self.n_node_type, self.n_edge_type, forward_type=self.diff_type,\
                            node_marginal=self.X_marginal, edge_marginal=self.E_marginal, device=self.device,
                            time_exponential=self.beta, time_base=self.alpha)
        
        example_data = val_set[0]
        X_t, E_t, node_mask, _ = to_dense(example_data)
        X_t, E_t, y_t = self.add_auxiliary_feature(X_t, E_t, node_mask)
        X_dim = X_t.shape[-1]
        E_dim = E_t.shape[-1]
        if self.extra_time:
            y_dim = y_t.shape[-1] + 1 # the extra time dimension
        else:
            y_dim = y_t.shape[-1]

        self.input_dims = {'X': X_dim, 'E': E_dim, 'y': y_dim }
        self.output_dims = {'X': self.n_node_type, 'E': self.n_edge_type, 'y': 0 }
        self.hidden_mlp_dims = {'X': self.n_dim, 'E': self.n_dim, 'y': self.n_dim }
        self.hidden_dims = {'dx': self.n_dim, 'de': self.n_dim, 'dy': self.n_dim, 'n_head': 8, 'dim_ffX': self.n_dim, 'dim_ffE': self.n_dim, 'dim_ffy': self.n_dim }

        if self.backbone == 'GT':
            if args.dataset in ['sbm', 'planar', 'community']:
                self.hidden_mlp_dims = { 'X': 128, 'E': 64, 'y': 128 }
                self.hidden_dims = { 'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256 }
            if args.dataset == 'moses':
                self.hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 256}
                self.hidden_dims = {'dx': 256, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 256}
            if args.dataset == 'guacamol':
                self.hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 256}
                self.hidden_dims = {'dx': 256, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 256}
            if args.dataset == "NAS-Bench-NLP":
                self.hidden_mlp_dims = {'X': 512, 'E': 256, 'y': 256}
                self.hidden_dims = {'dx': 512, 'de': 256, 'dy': 256, 'n_head': 8, 'dim_ffX': 512, 'dim_ffE': 256, 'dim_ffy': 256}
            if args.dataset in {"Younger", "YoungerPart"}:
                self.hidden_mlp_dims = {'X': 512, 'E': 256, 'y': 256}
                self.hidden_dims = {'dx': 512, 'de': 256, 'dy': 256, 'n_head': 8, 'dim_ffX': 512, 'dim_ffE': 256, 'dim_ffy': 256}
            self.model = GraphTransformer(n_layers=args.n_layers,
                input_dims=self.input_dims, 
                hidden_mlp_dims=self.hidden_mlp_dims,
                hidden_dims=self.hidden_dims,
                output_dims=self.output_dims).to(self.device)
        elif args.backbone == 'MPNN':
            self.model = MPNN(n_layers=self.n_layers,
                input_dims=self.input_dims,
                hidden_dims=self.n_dim,
                output_dims=self.output_dims).to(self.device)

        self.logger.info(self.model)
        self.logger.info("# of parameters: {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8)
        self.sampler = TauLeaping(self.n_node_type,
                self.n_edge_type,
                num_steps=self.sampling_steps,
                min_t=self.min_time,
                add_auxiliary_feature=self.add_auxiliary_feature,
                device=self.device,
                BAR=self.BAR, 
                level_fea=self.level_fea)
        
        self.estimator = Estimator(
            self.n_node_type, 
            self.n_edge_type,
            num_steps=self.sampling_steps,
            min_t=self.min_time,
            add_auxiliary_feature=self.add_auxiliary_feature,
            device=self.device,
            BAR=False,
            level_fea=self.level_fea,
            extra_time=self.extra_time
        )


    @overrides
    def train_step(self, data):
        include_node_feature = self.diffuser.diffuse_node
        data = data.to(self.device)
        X_0, E_0, node_mask, level = to_dense(data)


        ts = torch.rand((E_0.shape[0],), device=self.device) * (1.0 - self.min_time) + self.min_time

        X_t_idx, E_t_idx = self.diffuser.forward_diffusion(X_0, E_0, ts) # the output of diffuser is index-based, not one hot.
        
        X_t_one_hot = F.one_hot(X_t_idx, num_classes=self.n_node_type).float()
        E_t_one_hot = F.one_hot(E_t_idx, num_classes=self.n_edge_type).float()
        X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
        if self.extra_time:
            y_t = torch.cat([y_t, ts.unsqueeze(-1)], dim=-1) # TODO

        pred_X_0, pred_E_0 = self.model(X_t, E_t, y_t, node_mask, level, self.level_fea)

        X_0_idx = torch.max(X_0, dim=-1)[1].long()
        E_0_idx = torch.max(E_0, dim=-1)[1].long()
        X_0_idx_masked, E_0_idx_masked = add_mask_idx(X_0_idx, E_0_idx, self.n_node_type, self.n_edge_type, node_mask)

        loss_E = self.train_loss(pred_E_0, E_0_idx_masked)
        if include_node_feature:
            loss_X = self.train_loss(pred_X_0, X_0_idx_masked)
        else:
            loss_X = torch.zeros_like(loss_E)
        
        loss_y = torch.zeros_like(loss_E)
        loss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y
        
        loss_dict = {
            "loss": loss,
            "loss_X": loss_X,
            "loss_E": loss_E,
            "loss_y": loss_y
        }

        return loss_dict

    @overrides
    def validation_step(self, data):
        raise NotImplementedError

    @overrides
    def test_step(self, data):
        include_node_feature = self.diffuser.diffuse_node
        data = data.to(self.device)
        X_0, E_0, node_mask, level = to_dense(data)
        ts = torch.rand((E_0.shape[0],), device=self.device) * (1.0 - self.min_time) + self.min_time

        X_t_idx, E_t_idx = self.diffuser.forward_diffusion(X_0, E_0, ts) # the output of diffuser is index-based, not one hot.
        
        X_t_one_hot = F.one_hot(X_t_idx, num_classes=self.n_node_type).float()
        E_t_one_hot = F.one_hot(E_t_idx, num_classes=self.n_edge_type).float()
        X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
        if self.extra_time:
            y_t = torch.cat([y_t, ts.unsqueeze(-1)], dim=-1) # TODO

        pred_X_0, pred_E_0 = self.model(X_t, E_t, y_t, node_mask, level, self.level_fea)

        X_0_idx = torch.max(X_0, dim=-1)[1].long()
        E_0_idx = torch.max(E_0, dim=-1)[1].long()
        X_0_idx_masked, E_0_idx_masked = add_mask_idx(X_0_idx, E_0_idx, self.n_node_type, self.n_edge_type, node_mask)

        return pred_X_0, pred_E_0, X_0_idx_masked, E_0_idx_masked

    @overrides
    def generate_step(self):
        self.model.eval()
        n_node = self.n_node_distribution.sample((self.n_sample,)).to(self.device)
        X, E, node_mask = self.sampler.sample(self.diffuser, self.model, n_node)
        return X, E, node_mask, n_node
    
    @overrides
    def estimate_step(self, data):
        data = data.to(self.device)
        logp_dict = self.estimator.logp(self.diffuser, self.model, data)
        return logp_dict

    @overrides
    def get_model(self):
        return self.model
    
    @overrides
    def get_optimizer(self):
        return self.optimizer

    @overrides
    def get_scheduler(self):
        if self.enable_scheduler:
            return self.scheduler
        else:
            return None
