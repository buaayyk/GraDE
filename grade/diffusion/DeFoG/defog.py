from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.utils.logger import get_logger

import torch
import types
from torch import nn
from overrides import overrides
from torch.nn import functional as F
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

from src.models.extra_features import DummyExtraFeatures, ExtraFeatures
from src.models.transformer_model import GraphTransformer
from src.metrics.train_metrics import TrainLossDiscrete
from src import utils
from src.flow_matching.noise_distribution import NoiseDistribution
from src.flow_matching.time_distorter import TimeDistorter
from src.flow_matching.rate_matrix import RateMatrixDesigner
from src.flow_matching.utils import p_xt_g_x1
from src.flow_matching import flow_matching_utils
from src.datasets.dataset_utils import DistributionNodes

class DeFoG(DiffusionAlgorithm):
    
    def __init__(self, args, train_set, val_set, test_set):
        super().__init__()
        """ ========================== General training settings ========================== """
        self.device = args.device
        self.batch_size = args.batch_size

        """ ========================== Diffusion settings ========================== """
        self.diff_type = args.diff_type
        self.sampling_steps = args.sampling_steps
        self.n_sample = args.n_sample
        self.conditional = args.conditional
        self.rdb = args.rdb
        self.rdb_crit = args.rdb_crit
        self.eta = args.eta
        self.omega = args.omega
        self.level_fea = args.level_fea

        """ ============ Parametric model settings ============="""
        self.lr = args.lr
        self.wd = args.wd
        self.lambda_train = args.lambda_train
        self.n_layers = args.n_layers
        self.extra_fea = args.extra_fea
        self.rrwp_steps = args.rrwp_steps
        
        """ ============= Time Distortion ======================"""
        self.train_time_distortion = args.train_time_distortion
        self.sample_time_distortion = args.sample_time_distortion

        self.device = 'cpu' if self.device == 'cpu' else 'cuda:' + self.device
        dataset_info_dict = train_set.get_dataset_info()
        self.dataset_info = types.SimpleNamespace(**dataset_info_dict)
        self.node_dist = DistributionNodes(torch.tensor(self.dataset_info.n_node_distribution).to(self.device).float())
        self.node_types = torch.tensor(self.dataset_info.X_marginal).to(self.device).float()
        self.edge_types = torch.tensor(self.dataset_info.E_marginal).to(self.device).float()
        self.dataset_info.node_dist = self.node_dist
        self.dataset_info.node_types = self.node_types
        self.dataset_info.edge_types = self.edge_types
        self.extra_features = ExtraFeatures(self.extra_fea, self.rrwp_steps, self.dataset_info) if self.extra_fea is not None else DummyExtraFeatures()
        self.domain_features = DummyExtraFeatures()
        self.logger = get_logger()

        self.logger.info("============= DeFoG General Settings =============")
        self.logger.info('GPU: {}'.format(self.device))
        self.logger.info("Dataset: {}".format(args.dataset)) # TODO: DeFoG shouldn't konw args.dataset
        self.logger.info("Batch size: {}".format(self.batch_size))
        self.logger.info("lr: {}".format(self.lr))
        self.logger.info("wd: {}".format(self.wd))
        self.logger.info("lambda_train: {}".format(self.lambda_train))
        self.logger.info("# of layers: {}".format(self.n_layers))
        self.logger.info("============= DeFoG Diffusion Settings =============")
        self.logger.info("Diffusion Type: {}".format(self.diff_type))
        self.logger.info("# of samples: {}".format(self.n_sample))
        self.logger.info("Sampling steps: {}".format(self.sampling_steps))
        self.logger.info('E marginal: {}'.format(self.edge_types.tolist()))
        self.logger.info('X marginal: {}'.format(self.node_types.tolist()))
        self.logger.info(f'Train Time Distortion: {self.train_time_distortion}')
        self.logger.info(f'Sample Time Distortion: {self.sample_time_distortion}')
        self.logger.info(f"Level Feature: {self.level_fea}")
        self.logger.info("")

        """ ========================== Models ========================== """
        example_data = train_set[0]
        ex_dense, node_mask = utils.to_dense(example_data.x, example_data.edge_index, example_data.edge_attr, example_data.batch)
        X_t = ex_dense.X
        E_t = ex_dense.E
        y_t = example_data.y
        example_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 'node_mask': node_mask}
        ex_extra_feat = self.extra_features(example_data)
        ex_extra_molecular_feat = self.domain_features(example_data)

        self.input_dims = {
            'X': X_t.shape[-1] + ex_extra_feat.X.shape[-1] + ex_extra_molecular_feat.X.shape[-1], 
            'E': E_t.shape[-1] + ex_extra_feat.E.shape[-1] + ex_extra_molecular_feat.E.shape[-1], 
            'y': y_t.shape[-1] + ex_extra_feat.y.shape[-1] + ex_extra_molecular_feat.y.shape[-1]
        } # "+1" for y because of torch.cat(y, t)
        self.output_dims = {'X': X_t.shape[-1], 'E': E_t.shape[-1], 'y': y_t.shape[-1]}
        self.dataset_info.input_dims = self.input_dims
        self.dataset_info.output_dims = self.output_dims
        self.hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
        self.hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}

        self.noise_dist = NoiseDistribution(self.diff_type, self.dataset_info)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        self.train_loss = TrainLossDiscrete(
            self.lambda_train,
        ).to(self.device)

        self.model = GraphTransformer(
            n_layers=self.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=self.hidden_mlp_dims,
            hidden_dims=self.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        ).to(self.device)
        self.logger.info(self.model)
        self.logger.info("# of parameters: {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=self.train_time_distortion,
            sample_distortion=self.sample_time_distortion,
            alpha=1,
            beta=1,
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.rdb,
            rdb_crit=self.rdb_crit,
            eta=self.eta,
            omega=self.omega,
            limit_dist=self.limit_dist,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.wd,
        )

        self.scheduler = None

    @overrides
    def get_model(self):
        return self.model

    @overrides    
    def get_optimizer(self):
        return self.optimizer
    
    @overrides
    def get_scheduler(self):
        return self.scheduler
    
    @overrides
    def train_step(self, data):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        self.train_loss.reset()
        data = data.to(self.device)

        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1
        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        if getattr(data, "level", None) is not None:
            level, _ = to_dense_batch(x=data.level, batch=data.batch)
        else:
            level = None
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, level)

        loss_dict = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y
        )

        return loss_dict
    
    @overrides
    def validation_step(self, data):
        return
    
    @overrides
    def test_step(self, data):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        data = data.to(self.device)

        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1

        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        if getattr(data, "level", None) is not None:
            level, _ = to_dense_batch(x=data.level, batch=data.batch)
        else:
            level = None
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, level)

        X_0_idx_masked = torch.argmax(dense_data.X, dim=-1).long()
        E_0_idx_masked = torch.argmax(dense_data.E, dim=-1).long()
        mask_X = (dense_data.X == 0).all(dim=-1)
        mask_E = (dense_data.E == 0).all(dim=-1)
        X_0_idx_masked[mask_X] = self.output_dims['X']
        E_0_idx_masked[mask_E] = self.output_dims['E']

        return pred.X, pred.E, X_0_idx_masked, E_0_idx_masked

    @overrides
    def generate_step(self):
        samples_left_to_generate = self.n_sample
        samples = []
        while samples_left_to_generate > 0:
            to_generate = min(samples_left_to_generate, self.batch_size)
            samples.extend(self.sample_batch(to_generate))
            samples_left_to_generate -= to_generate
        return samples

    @overrides
    def estimate_step(self, data):
        data = data.to(self.device)
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        edge_mask = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2))
        dense_data = dense_data.mask(node_mask)
        bs, n_node = dense_data.X.shape[0], dense_data.X.shape[1]
        n_node_type = dense_data.X.shape[-1]
        n_edge_type = dense_data.E.shape[-1]
        limit_X = self.limit_dist.X.to(self.device).float()
        limit_E = self.limit_dist.E.to(self.device).float()
        z_s = utils.PlaceHolder(X=dense_data.X, E=dense_data.E, y=data.y).type_as(dense_data.X).mask(node_mask)
        X_s, E_s, y = z_s.X, z_s.E, z_s.y
        if getattr(data, "level", None) is not None:
            level, _ = to_dense_batch(x=data.level, batch=data.batch)
        else:
            level = None
        X_logp = torch.zeros((bs, n_node), device=self.device).float()
        E_logp = torch.zeros((bs, n_node, n_node), device=self.device).float()

        for s_int in tqdm(range(self.sampling_steps, 0, -1)):
            s_array = s_int * torch.ones((bs, 1)).float().to(self.device)
            s_norm = s_array / (self.sampling_steps)
            t_array = s_array - 1
            t_norm = t_array / (self.sampling_steps)
            if ("absorb" in self.diff_type) and (s_int - 1 == 0):
                t_norm = t_norm + 1e-6

            # Distort time
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.sample_time_distortion
            )
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.sample_time_distortion
            )

            """========= Noising Process =========="""
            X_pts = (t_norm / s_norm).unsqueeze(-1) * torch.eye(n_node_type).unsqueeze(0).expand(bs, -1, -1).float().to(self.device) + (1 - t_norm / s_norm).unsqueeze(-1) * limit_X.unsqueeze(0).unsqueeze(0).expand(bs, n_node_type, -1) # (bs, n_node_type, n_node_type), p_{t|s}(.|X_s)
            E_pts = (t_norm / s_norm).unsqueeze(-1) * torch.eye(n_edge_type).unsqueeze(0).expand(bs, -1, -1).float().to(self.device) + (1 - t_norm / s_norm).unsqueeze(-1) * limit_E.unsqueeze(0).unsqueeze(0).expand(bs, n_edge_type, -1) # (bs, n_edge_type, n_edge_type), p_{t|s}(.|E_s)
            X_s_idx = torch.argmax(X_s, dim=-1).long()
            E_s_idx = torch.argmax(E_s, dim=-1).long()
            X_t_prob = X_pts.gather(dim=1, index=X_s_idx.unsqueeze(-1).expand(bs, n_node, n_node_type)) # (bs, n_node, n_node_type)
            E_t_prob = E_pts.gather(dim=1, index=E_s_idx.view(bs, -1).unsqueeze(-1).expand(bs, n_node * n_node, n_edge_type)).view(bs, n_node, n_node, n_edge_type) # (bs, n_node, n_node, n_edge_type)
            sampled_t = flow_matching_utils.sample_discrete_features(probX=X_t_prob, probE=E_t_prob, node_mask=node_mask)
            X_t = F.one_hot(sampled_t.X, num_classes=n_node_type)
            E_t = F.one_hot(sampled_t.E, num_classes=n_edge_type)
            X_t_idx = torch.argmax(X_t, dim=-1).long()
            E_t_idx = torch.argmax(E_t, dim=-1).long()
            noising_X_probs = X_t_prob.gather(dim=-1, index=X_t_idx.unsqueeze(-1)).squeeze(-1)
            noising_E_probs = E_t_prob.gather(dim=-1, index=E_t_idx.unsqueeze(-1)).squeeze(-1)
            noising_X_logp = torch.log(noising_X_probs) * node_mask # (bs, n_node), \log{p_{t|s}(X_t|X_s)}
            noising_E_logp = torch.log(noising_E_probs) * edge_mask # (bs, n_node, n_node), \log{p_{t|s}(E_t | E_s)}

            """========= Denoising Process =========="""
            noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y, 't': t_norm, 'node_mask': node_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask, level)
            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)  # bs, n_node, n_node_type
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n_node, n_node, n_edge_type
            G_t = X_t, E_t
            G_1_pred = pred_X, pred_E

            dfm_variables = self.compute_dfm_variables(t_norm, X_t_idx, E_t_idx)
            Rstar_X, Rstar_E = self.compute_Rstar(dfm_variables)
            Rdb_X, Rdb_E = self.compute_Rdb(dfm_variables)
            Rtg_X, Rtg_E = self.compute_Rtg(dfm_variables)

            R_t_X = Rstar_X + Rdb_X + Rtg_X # (bs, n_node, j: n_node_type, k: n_node_type)
            R_t_E = Rstar_E + Rdb_E + Rtg_E # (bs, n_node, n_node, j: n_edge_type, k: n_edge_type)
            R_t_X = torch.nan_to_num(R_t_X, nan=0.0, posinf=0.0, neginf=0.0)
            R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0)
            R_t_X = torch.einsum("bnjk, bnk->bnj", R_t_X, pred_X) # (bs, n_node, n_node_type)
            R_t_E = torch.einsum("bmnjk, bmnk->bmnj", R_t_E, pred_E) # (bs, n_node, n_node, n_edge_type)

            R_t_X, R_t_E = self.postprocess_R(R_t_X, R_t_E, X_t_idx, E_t_idx)

            R_t_X_ts = torch.gather(R_t_X, index=X_s_idx.unsqueeze(-1), dim=-1).squeeze(-1) # (bs, n_node) R_{t|1}(x_{t}, x_{t+\Delta t})
            R_t_X_tt = torch.gather(R_t_X, index=X_t_idx.unsqueeze(-1), dim=-1).squeeze(-1) # (bs, n_node) R_{t|1}(x_{t}, x_{t})

            R_t_E_ts = torch.gather(R_t_E, index=E_s_idx.unsqueeze(-1), dim=-1).squeeze(-1) # (bs, n_node, n_node) R_{t|1}(e_{t}, e_{t+\Delta t})
            R_t_E_tt = torch.gather(R_t_E, index=E_t_idx.unsqueeze(-1), dim=-1).squeeze(-1) # (bs, n_node, n_node) R_{t|1}(e_{t}, e_{t})

            h = s_norm - t_norm
            denoising_X_probs = (1 - torch.exp(h.view(bs, 1) * R_t_X_tt)) * R_t_X_ts / (-R_t_X_tt).clamp_min(1e-12) * (X_s_idx != X_t_idx) + torch.exp(h.view(bs, 1) * R_t_X_tt) * (X_s_idx == X_t_idx)
            denoising_E_probs = (1 - torch.exp(h.view(bs, 1, 1) * R_t_E_tt)) * R_t_E_ts / (-R_t_E_tt).clamp_min(1e-12) * (E_s_idx != E_t_idx) + torch.exp(h.view(bs, 1, 1) * R_t_E_tt) * (E_s_idx == E_t_idx)
            denoising_X_probs = denoising_X_probs.clamp_min(1e-6)
            denoising_E_probs = denoising_E_probs.clamp_min(1e-6)

            # R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            #     t_norm,
            #     node_mask,
            #     G_t,
            #     G_1_pred,
            # )
            # X_s_prob, E_s_prob = self.compute_step_probs(
            #     R_t_X, R_t_E, X_t, E_t, (s_norm - t_norm)[0], self.limit_dist.X, self.limit_dist.E
            # ) # (bs, n_node, n_node_type), (bs, n_node, n_node, n_edge_type)
            # denoising_X_probs = X_s_prob.gather(dim=-1, index=X_s_idx.unsqueeze(-1)).squeeze(-1)
            # denoising_E_probs = E_s_prob.gather(dim=-1, index=E_s_idx.unsqueeze(-1)).squeeze(-1)

            denoising_X_logp = torch.log(denoising_X_probs) # (bs, n_node), \log{p_{s|t}(X_s|G_t)}
            denoising_E_logp = torch.log(denoising_E_probs) # (bs, n_node, n_node), \log{p_{s|t}(E_s|G_t)}

            """========== Update logp =============="""
            X_logp += (denoising_X_logp - noising_X_logp)
            E_logp += (denoising_E_logp - noising_E_logp)
            X_s, E_s = X_t, E_t
        
        X_ref_distribution = limit_X.view(1, 1, -1).expand(bs, n_node, -1).to(self.device)
        E_ref_distribution = limit_E.view(1, 1, 1, -1).expand(bs, n_node, n_node, -1).to(self.device)
        ref_X_probs = torch.gather(X_ref_distribution, -1, torch.argmax(X_t, dim=-1, keepdim=True)).squeeze(-1)
        ref_E_probs = torch.gather(E_ref_distribution, -1, torch.argmax(E_t, dim=-1, keepdim=True)).squeeze(-1)
        ref_X_logp = torch.log(ref_X_probs) # (bs, n_node, n_node_type), \pi(X_0)
        ref_E_logp = torch.log(ref_E_probs) # (bs, n_node, n_node, n_edge_type), \pi(E_0)
        # X_logp += ref_X_logp
        # E_logp += ref_E_logp

        X_logp = torch.sum(X_logp * node_mask, dim=1)
        E_logp = torch.sum((E_logp * edge_mask).triu(diagonal=1), dim=[1, 2])
        G_logp = X_logp + E_logp
        
        return {
            "G_logp": G_logp, 
            "X_logp": X_logp, 
            "E_logp": E_logp
        }
        
    def compute_dfm_variables(self, t_norm, X_t_idx, E_t_idx):
        # R_{t|1}(i, j | k)
        # i <--> x_t, e_t  |  j <--> x_{t+\Delta t}, e_{t+\Delta t} | k <--> x_1, e_1
        
        bs, n_node = X_t_idx.shape[0], X_t_idx.shape[1] 
        n_node_type, n_edge_type = self.output_dims['X'], self.output_dims['E']
        limit_X, limit_E = self.limit_dist.X, self.limit_dist.E
        t = t_norm.view(bs, 1, 1, 1)
           
        X_delta_j_k = torch.eye(n_node_type, device=self.device).float().view(1, 1, n_node_type, n_node_type).expand(bs, n_node, -1, -1) # (bs, n_node, j: n_node_type, k: n_node_type) \delta(x_{t+\Delta}, x_1)
        X_delta_i_k = F.one_hot(X_t_idx, num_classes=n_node_type).float().view(bs, n_node, 1, n_node_type).expand(-1, -1, n_node_type, -1) # (bs, n_node, j: n_node_type, k: n_node_type) \delta(x_t, x_1), fix x_t
        X_p0_j = limit_X.view(1, 1, n_node_type, 1).expand(bs, n_node, -1, n_node_type) # (bs, n_node, j: n_node_type, k: n_node_type) p_0(x_{t+\Delta t})
        X_p0_i = limit_X[X_t_idx].view(bs, n_node, 1, 1).expand(-1, -1, n_node_type, n_node_type) # (bs, n_node, j: n_node_type, k: n_node_type) p_0(x_t), fix x_t
        # Rstar_numer_X = ReLU[ (\delta(x_{t+\Delta}, x_1) - p_0(x_{t+\Delta t})) - (\delta(x_t, x_1) - p_0(x_t)) ]
        # Rstar_numer_X = F.relu((X_delta_j_k - X_p0_j) - (X_delta_i_k - X_p0_i))

        X_ptg1_i_k = t * X_delta_i_k + (1 - t) * X_p0_i # (bs, n_node, j: n_node_type, k: n_node_type) p_{t|1}(x_t|x_1), fix x_t
        X_ptg1_j_k = t * X_delta_j_k + (1 - t) * X_p0_j # (bs, n_node, j: n_node_type, k: n_node_type) p_{t|1}(x_{t+\Delta t}|x_1)
        X_delta_all_i_k = torch.eye(n_node_type, device=self.device).float().view(1, 1, n_node_type, n_node_type).expand(bs, n_node, -1, -1) # (bs, n_node, i: n_node_type, k: n_node_type) \delta(., x_1), not fix x_t
        X_p0_all_i = limit_X.view(1, 1, n_node_type, 1).expand(bs, n_node, -1, n_node_type)  # (bs, n_node, i: n_node_type, k: n_node_type) p_0(.), not fix x_t
        X_ptg1_all_i_k = t * X_delta_all_i_k + (1.0 - t) * X_p0_all_i  # (bs, n_node, i: n_node_type, k: n_node_type) p_{t|1}(. | x_1), not fix x_t
        Z_t_X = torch.count_nonzero(X_ptg1_all_i_k, dim=2).float().unsqueeze(2).expand(-1, -1, n_node_type, -1)  # (bs, n_node, j: n_node_type, k: n_node_type)
        # Rstar_denom_E = Z_t^{>0} * p_{t|1}(x_t | x_1)
        # Rstar_denom_X = Z_t_X * X_ptg1_i_k # (bs, n_node, j: n_node_type, k: n_node_type)

        E_delta_j_k = torch.eye(n_edge_type, device=self.device).float().view(1, 1, n_edge_type, n_edge_type).expand(bs, n_node * n_node, -1, -1) # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) \delta(e_{t+\Delta}, e_1)
        E_delta_i_k = F.one_hot(E_t_idx, num_classes=n_edge_type).float().view(bs, n_node * n_node, 1, n_edge_type).expand(-1, -1, n_edge_type, -1) # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) \delta(e_t, e_1), fix e_t
        E_p0_j = limit_E.view(1, 1, n_edge_type, 1).expand(bs, n_node * n_node, -1, n_edge_type) # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) p_0(e_{t+\Delta t})
        E_p0_i = limit_E[E_t_idx].view(bs, n_node * n_node, 1, 1).expand(-1, -1, n_edge_type, n_edge_type) # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) p_0(e_t), fix e_t
        # Rstar_numer_E = ReLU[ (\delta(e_{t+\Delta}, e_1) - p_0(e_{t+\Delta t})) - (\delta(e_t, e_1) - p_0(e_t)) ]
        # Rstar_numer_E = F.relu((E_delta_j_k - E_p0_j) - (E_delta_i_k - E_p0_i))

        E_ptg1_i_k = t * E_delta_i_k + (1 - t) * E_p0_i # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) p_{t|1}(e_t|e_1), fix e_t
        E_ptg1_j_k = t * E_delta_j_k + (1 - t) * E_p0_j # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) p_{t|1}(e_{t+\Delta t}|e_1)
        E_delta_all_i_k = torch.eye(n_edge_type, device=self.device).float().view(1, 1, n_edge_type, n_edge_type).expand(bs, n_node * n_node, -1, -1) # (bs, n_node * n_node, i: n_edge_type, k: n_edge_type) \delta(., e_1), not fix e_t
        E_p0_all_i = limit_E.view(1, 1, n_edge_type, 1).expand(bs, n_node * n_node, -1, n_edge_type)  # (bs, n_node * n_node, i: n_edge_type, k: n_edge_type) p_0(.), not fix e_t
        E_ptg1_all_i_k = t * E_delta_all_i_k + (1.0 - t) * E_p0_all_i  # (bs, n_node * n_node, i: n_edge_type, k: n_edge_type) p_{t|1}(. | e_1), not fix e_t
        Z_t_E = torch.count_nonzero(E_ptg1_all_i_k, dim=2).float().unsqueeze(2).expand(-1, -1, n_edge_type, -1)  # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type)
        # Rstar_denom_E = Z_t^{>0} * p_{t|1}(e_t | e_1)        
        # Rstar_denom_E = Z_t_E * E_ptg1_i_k # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type)

        return {
            "X_delta_j_k": X_delta_j_k,
            "X_delta_i_k": X_delta_i_k,
            "X_p0_j": X_p0_j,
            "X_p0_i": X_p0_i,
            "X_ptg1_i_k": X_ptg1_i_k,
            "X_ptg1_j_k": X_ptg1_j_k,
            "Z_t_X": Z_t_X,
            "E_delta_j_k": E_delta_j_k,
            "E_delta_i_k": E_delta_i_k,
            "E_p0_j": E_p0_j,
            "E_p0_i": E_p0_i,
            "E_ptg1_i_k": E_ptg1_i_k,
            "E_ptg1_j_k": E_ptg1_j_k,
            "Z_t_E": Z_t_E
        }

    def compute_Rstar(self, dfm_variables):
        # R_{t|1}(i, j | k)
        # i <--> x_t, e_t  |  j <--> x_{t+\Delta t}, e_{t+\Delta t} | k <--> x_1, e_1
        
        X_delta_j_k = dfm_variables["X_delta_j_k"]
        X_p0_j = dfm_variables["X_p0_j"]
        X_delta_i_k = dfm_variables["X_delta_i_k"]
        X_p0_i = dfm_variables["X_p0_i"]
        Z_t_X = dfm_variables["Z_t_X"]
        X_ptg1_i_k = dfm_variables["X_ptg1_i_k"]

        E_delta_j_k = dfm_variables["E_delta_j_k"]
        E_p0_j = dfm_variables["E_p0_j"]
        E_delta_i_k = dfm_variables["E_delta_i_k"]
        E_p0_i = dfm_variables["E_p0_i"]
        Z_t_E = dfm_variables["Z_t_E"]
        E_ptg1_i_k = dfm_variables["E_ptg1_i_k"]

        bs, n_node = X_delta_j_k.shape[0], X_delta_j_k.shape[1]
        n_node_type, n_edge_type = X_delta_j_k.shape[2], E_delta_j_k.shape[2]

        Rstar_numer_X = F.relu((X_delta_j_k - X_p0_j) - (X_delta_i_k - X_p0_i)) # (bs, n_node, j: n_node_type, k: n_node_type)
        Rstar_denom_X = Z_t_X * X_ptg1_i_k # (bs, n_node, j: n_node_type, k: n_node_type)
        Rstar_X = Rstar_numer_X / Rstar_denom_X # (bs, n_node, j: n_node_type, k: n_node_type)

        Rstar_numer_E = F.relu((E_delta_j_k - E_p0_j) - (E_delta_i_k - E_p0_i)) # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type)
        Rstar_denom_E = Z_t_E * E_ptg1_i_k # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type)
        Rstar_E = (Rstar_numer_E / Rstar_denom_E).view(bs, n_node, n_node, n_edge_type, n_edge_type) # (bs, n_node, n_node, j: n_edge_type, k: n_edge_type)

        return Rstar_X, Rstar_E
    
    def compute_Rdb(self, dfm_variables):
        # R_{t|1}(i, j | k)
        # i <--> x_t, e_t  |  j <--> x_{t+\Delta t}, e_{t+\Delta t} | k <--> x_1, e_1

        if self.rdb == 'general':
            pass
        else:
            raise NotImplementedError

        X_ptg1_j_k = dfm_variables["X_ptg1_j_k"]
        E_ptg1_j_k = dfm_variables["E_ptg1_j_k"]

        bs, n_node = X_ptg1_j_k.shape[0], X_ptg1_j_k.shape[1]
        n_node_type, n_edge_type = X_ptg1_j_k.shape[2], E_ptg1_j_k.shape[2]

        Rdb_X = self.eta * X_ptg1_j_k # (bs, n_node, j: n_node_type, k: n_node_type) \eta * p_{t|1}(x_{t+\Delta t}|x_1)
        Rdb_E = self.eta * E_ptg1_j_k
        Rdb_E = Rdb_E.view(bs, n_node, n_node, n_edge_type, n_edge_type) # (bs, n_node, n_node, n_edge_type, n_edge_type) \eta * p_{t|1}(e_{t+\Delta t}|e_1)

        return Rdb_X, Rdb_E

    def compute_Rtg(self, dfm_variables):
        # R_{t|1}(i, j | k)
        # i <--> x_t, e_t  |  j <--> x_{t+\Delta t}, e_{t+\Delta t} | k <--> x_1, e_1
        
        X_delta_j_k = dfm_variables["X_delta_j_k"]
        Z_t_X = dfm_variables["Z_t_X"]
        X_ptg1_i_k = dfm_variables["X_ptg1_i_k"]
        X_delta_i_k = dfm_variables["X_delta_i_k"]

        E_delta_j_k = dfm_variables["E_delta_j_k"]
        Z_t_E = dfm_variables["Z_t_E"]
        E_ptg1_i_k = dfm_variables["E_ptg1_i_k"]
        E_delta_i_k = dfm_variables["E_delta_i_k"]

        bs, n_node = X_delta_j_k.shape[0], X_delta_j_k.shape[1]
        n_node_type, n_edge_type = X_delta_j_k.shape[2], E_delta_j_k.shape[2]

        # Rtg(i, j | k) = \omega * \cfrac{\delta(j, k) * (1 - \delta(i, j))}{Z * p_{t|1}(i | k)}
        # Because \delta(j, k) * (1 - \delta(i, j) = \delta(j, k) * (1 - \delta(i, k))
        # So X_mask_non_self / E_mask_non_self = 1 - \delta(i, j) = 1 - \delta(i, k)

        X_mask_non_self = 1.0 - X_delta_i_k # (bs, n_node, j: n_node_type, k: n_node_type), 1 - \delta(i, k)
        E_mask_non_self = 1.0 - E_delta_i_k # (bs, n_node * n_node, j: n_edge_type, k: n_edge_type) 

        Rtg_X = self.omega * X_delta_j_k * X_mask_non_self / (Z_t_X * X_ptg1_i_k) # (bs, n_node, j: n_node_type, k: n_node_type)
        Rtg_E = self.omega * E_delta_j_k * E_mask_non_self / (Z_t_E * E_ptg1_i_k)
        Rtg_E = Rtg_E.view(bs, n_node, n_node, n_edge_type, n_edge_type) # (bs, n_node, n_node, j: n_edge_type, k: n_edge_type)
        
        return Rtg_X, Rtg_E

    def postprocess_R(self, R_t_X, R_t_E, X_t_idx, E_t_idx):
        n_node_type = R_t_X.shape[-1]
        n_edge_type = R_t_E.shape[-1]
        epsilon = 1e-6

        R_t_X = torch.nan_to_num(R_t_X, nan=0.0, posinf=0.0, neginf=0.0) # (bs, n_node, n_node_type)
        R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0) # (bs, n_node, n_node, n_edge_type)

        R_t_X = R_t_X + epsilon        
        diag_X = F.one_hot(X_t_idx, num_classes=n_node_type).float()            # (bs, n_node, n_node_type)
        offdiag_X = R_t_X * (1 - diag_X) # (bs, n_node, n_node_type)
        diag_val_X = -offdiag_X.sum(dim=-1, keepdim=True) # (bs, n_node, 1)
        R_t_X = offdiag_X + diag_val_X * diag_X # (bs, n_node, n_node_type)
        
        R_t_E = R_t_E + epsilon
        diag_E = F.one_hot(E_t_idx, num_classes=n_edge_type).float()            # (bs, n_node, n_node, n_edge_type)
        offdiag_E = R_t_E * (1 - diag_E) # (bs, n_node, n_node, n_edge_type)
        diag_val_E = -offdiag_E.sum(dim=-1, keepdim=True) # (bs, n_node, n_node, 1)
        R_t_E = offdiag_E + diag_val_E * diag_E # (bs, n_node, n_node, n_edge_type)
        
        return R_t_X, R_t_E
    
    def apply_noise(self, X, E, y, node_mask, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask, level=None):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        # return self.model(X, E, y, node_mask)
        return self.model(X, E, noisy_data['y_t'].float(), node_mask, level, self.level_fea)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        num_nodes=None
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
        )
        # X, E, y = z_T.X, z_T.E, z_T.y
        # TODO, y = ?
        X, E, y = z_T.X, z_T.E, torch.zeros((batch_size, 1)).float().to(self.device)

        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()

        for t_int in tqdm(range(0, self.sampling_steps)):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / (self.sampling_steps)
            if ("absorb" in self.diff_type) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / (self.sampling_steps)

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.sample_time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.sample_time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            # X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
    

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        return molecule_list

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        if self.conditional:
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            noisy_data["y_t"] = uncond_y

            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            R_t_X_uncond, R_t_E_uncond = (
                self.rate_matrix_designer.compute_graph_rate_matrix(
                    t,
                    node_mask,
                    G_t,
                    G_1_pred,
                )
            )

            guidance_weight = self.cfg.general.guidance_weight
            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 1], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)