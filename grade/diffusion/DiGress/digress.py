from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.utils.logger import get_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import random
import tqdm
from overrides import overrides

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.distributions import DistributionNodes

class DiGress(DiffusionAlgorithm):

    def __init__(self, args, train_set, val_set, test_set):
        super().__init__()
        """ ========================== General training settings ========================== """
        self.device = args.device
        self.batch_size = args.batch_size

        """ ========================== Diffusion settings ========================== """
        self.diff_type = args.diff_type
        self.sampling_steps = args.sampling_steps
        self.diffusion_noise_schedule = args.diffusion_noise_schedule
        self.n_sample = args.n_sample
        self.p_uncondition = args.p_uncondition
        self.gamma = args.gamma
        self.conditional = args.conditional
        self.use_full_y = bool(args.use_full_y)
        self.use_sin_pe = bool(args.use_sin_pe)

        """ ============ Parametric model settings ============="""
        self.lr = args.lr
        self.wd = args.wd
        self.lambda_train = args.lambda_train
        self.n_layers = args.n_layers


        self.device = 'cpu' if self.device == 'cpu' else 'cuda:' + self.device
        self.extra_features = DummyExtraFeatures()
        self.domain_features = DummyExtraFeatures()
        self.dataset_info = train_set.get_dataset_info()        
        self.node_dist = DistributionNodes(torch.tensor(self.dataset_info["n_node_distribution"]).to(self.device).float())
        self.X_marginal, self.E_marginal = self.dataset_info["X_marginal"], self.dataset_info["E_marginal"]
        self.E_marginal = torch.tensor(self.E_marginal).float().to(self.device)
        self.X_marginal = torch.tensor(self.X_marginal).float().to(self.device)
        self.logger = get_logger()

        self.logger.info("============= DiGress General Settings =============")
        self.logger.info('GPU: {}'.format(self.device))
        self.logger.info("Dataset: {}".format(args.dataset)) # TODO: DisCo shouldn't konw args.dataset
        self.logger.info("Batch size: {}".format(self.batch_size))
        self.logger.info("lr: {}".format(self.lr))
        self.logger.info("wd: {}".format(self.wd))
        self.logger.info("lambda_train: {}".format(self.lambda_train))
        self.logger.info("# of layers: {}".format(self.n_layers))
        self.logger.info("============= DisCo Diffusion Settings =============")
        self.logger.info("Diffusion Type: {}".format(self.diff_type))
        self.logger.info("# of samples: {}".format(self.n_sample))
        self.logger.info("Sampling steps: {}".format(self.sampling_steps))
        self.logger.info('E marginal: {}'.format(self.E_marginal.tolist()))
        self.logger.info('X marginal: {}'.format(self.X_marginal.tolist()))
        self.logger.info("Use full y: {}".format(self.use_full_y))
        self.logger.info("Use SinPE: {}".format(self.use_sin_pe))
        self.logger.info("")
        
        """ ========================== Losses ========================== """
        self.train_loss = TrainLossDiscrete(self.lambda_train).to(self.device)

        self.val_nll = NLL().to(self.device)
        self.val_X_kl = SumExceptBatchKL().to(self.device)
        self.val_E_kl = SumExceptBatchKL().to(self.device)
        self.val_X_logp = SumExceptBatchMetric().to(self.device)
        self.val_E_logp = SumExceptBatchMetric().to(self.device)

        self.test_nll = NLL().to(self.device)
        self.test_X_kl = SumExceptBatchKL().to(self.device)
        self.test_E_kl = SumExceptBatchKL().to(self.device)
        self.test_X_logp = SumExceptBatchMetric().to(self.device)
        self.test_E_logp = SumExceptBatchMetric().to(self.device)
        
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
            'y': (
                y_t.shape[-1] + ex_extra_feat.y.shape[-1] + ex_extra_molecular_feat.y.shape[-1] + 1
                if self.use_full_y else y_t.shape[-1]
            )
        } # Full-y mode appends extra y features and diffusion time t.
        self.output_dims = {'X': X_t.shape[-1], 'E': E_t.shape[-1], 'y': y_t.shape[-1]}
        self.Xdim, self.Edim, self.ydim = self.input_dims['X'], self.input_dims['E'], self.input_dims['y']
        self.Xdim_output, self.Edim_output, self.ydim_output = self.output_dims['X'], self.output_dims['E'], self.output_dims['y']
        self.hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
        self.hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}

        self.model = GraphTransformer(n_layers=self.n_layers,
                                      input_dims=self.input_dims,
                                      hidden_mlp_dims=self.hidden_mlp_dims,
                                      hidden_dims=self.hidden_dims,
                                      output_dims=self.output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU(),
                                      use_sin_pe=self.use_sin_pe).to(self.device)
        self.logger.info(self.model)
        self.logger.info("# of parameters: {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(self.diffusion_noise_schedule, timesteps=self.sampling_steps)

        if self.diff_type == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.output_dims['X'], e_classes=self.output_dims['E'], y_classes=self.output_dims['y'])
            x_limit = torch.ones(self.output_dims['X']) / self.output_dims['X']
            e_limit = torch.ones(self.output_dims['E']) / self.output_dims['E']
            y_limit = torch.ones(self.output_dims['y']) / self.output_dims['y']
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif self.diff_type == 'marginal':            
            self.transition_model = MarginalUniformTransition(x_marginals=self.X_marginal, e_marginals=self.E_marginal, y_classes=self.output_dims['y'])
            self.limit_dist = utils.PlaceHolder(X=self.X_marginal, E=self.E_marginal, y=torch.ones(self.output_dims['y']) / self.output_dims['y'])

        self.optimizer =  torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, verbose=True)
 

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
            print("Found a batch with no edges. Skipping.")
            return
        self.train_loss.reset()
        data = data.to(self.device)
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        
        is_class_cond = torch.rand(size=data.y.shape, device=data.x.device) >= self.p_uncondition
        data.y = data.y * is_class_cond.float()

        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)
        loss_dict = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y, true_X=X, true_E=E, true_y=data.y)

        return loss_dict

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()

    @overrides
    def validation_step(self, data):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        is_class_cond = torch.rand(size=data.y.shape, device=data.x.device) >= self.p_uncondition
        data.y = data.y * is_class_cond.float()

        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        return {'loss': nll}

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
    def test_step(self, data):
        data = data.to(self.device)
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        
        is_class_cond = torch.rand(size=data.y.shape, device=data.x.device) >= self.p_uncondition
        data.y = data.y * is_class_cond.float()
        
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)
        # nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)

        X_0_idx_masked = torch.argmax(dense_data.X, dim=-1).long()
        E_0_idx_masked = torch.argmax(dense_data.E, dim=-1).long()
        mask_X = (dense_data.X == 0).all(dim=-1)
        mask_E = (dense_data.E == 0).all(dim=-1)
        X_0_idx_masked[mask_X] = self.output_dims['X']
        E_0_idx_masked[mask_E] = self.output_dims['E']

        return pred.X, pred.E, X_0_idx_masked, E_0_idx_masked

    @overrides
    def estimate_step(self, data):
        # sample z1, .., zT according to z0, and then calculate log p(z0 | N)
        # To simplify, the variable's name in this function ignore 'given_N'
        data = data.to(self.device)
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        bs, n = dense_data.X.shape[0], dense_data.X.shape[1]
        z_s = utils.PlaceHolder(X=dense_data.X, E=dense_data.E, y=data.y).type_as(dense_data.X).mask(node_mask)
        
        log_p_X0 = torch.zeros((bs, n, 1), dtype=torch.float).to(dense_data.X.device)
        log_p_E0 = torch.zeros((bs, n, n, 1), dtype=torch.float).to(dense_data.X.device)

        for s_int in tqdm.tqdm(range(self.sampling_steps)):
            s_int = torch.tensor([s_int], device=z_s.X.device).float().unsqueeze(0).expand(z_s.X.size(0), -1)
            t_int = s_int + 1
            t_float = t_int / self.sampling_steps
            s_float = s_int / self.sampling_steps

            # 1. forward process
            # beta_t and alpha_s_bar are used for denoising/loss computation
            beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

            Qt = self.transition_model.get_Qt(beta_t, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)

            assert (abs(Qt.X.sum(dim=2) - 1.) < 1e-4).all(), Qt.X.sum(dim=2) - 1
            assert (abs(Qt.E.sum(dim=2) - 1.) < 1e-4).all()

            # Compute transition probabilities
            probX_t = z_s.X @ Qt.X  # (bs, n, dx_out)
            probE_t = z_s.E @ Qt.E.unsqueeze(1)  # (bs, n, n, de_out)
            sampled_t = diffusion_utils.sample_discrete_features(probX=probX_t, probE=probE_t, node_mask=node_mask)
            X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
            E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
            assert (z_s.X.shape == X_t.shape) and (z_s.E.shape == E_t.shape)
            z_t = utils.PlaceHolder(X=X_t, E=E_t, y=z_s.y).type_as(z_s.X).mask(node_mask)

            # q(z_t|z_s)
            p_Xt_given_zs = torch.gather(probX_t, -1, torch.argmax(z_t.X, dim=-1, keepdim=True))
            log_p_Xt_given_zs = torch.log(p_Xt_given_zs)
            p_Et_given_zs = torch.gather(probE_t, -1, torch.argmax(z_t.E, dim=-1, keepdim=True))
            log_p_Et_given_zs = torch.log(p_Et_given_zs)            

            # p(z_s|z_t)
            y_classes = self.output_dims['y']
            y_t = torch.tensor([1 if i == y_classes - 1 else 0 for i in range(y_classes)]).repeat(z_t.y.shape[0], 1).to(self.device)
            
            if self.conditional is False:
                y_t = y_t * 0

            noisy_data = {}

            noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t_float, 'node_mask': node_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask) # p(z0|zt)
            
            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

            p_s_given_t_and_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.X, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X) # p(Xs | zt, z0)
            p_s_given_t_and_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t, Qt=Qt.E,Qsb=Qsb.E, Qtb=Qtb.E) # p(Es | zt, z0)

            # Dim of these two tensors: bs, N, d0, d_t-1
            weighted_X = pred_X.unsqueeze(-1) * p_s_given_t_and_0_X         # bs, n, d0, d_t-1
            unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            probX_s = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

            pred_E = pred_E.reshape((pred_E.shape[0], -1, pred_E.shape[-1])) # bs, n * n, d_t-1
            weighted_E = pred_E.unsqueeze(-1) * p_s_given_t_and_0_E        # bs, n * n, d0, d_t-1
            unnormalized_prob_E = weighted_E.sum(dim=-2) # bs, n * n, d_t-1
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            probE_s = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True) # bs, n * n, d_t-1
            probE_s = probE_s.reshape(bs, n, n, -1) # bs, n, n, d_t-1

            assert ((probX_s.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((probE_s.sum(dim=-1) - 1).abs() < 1e-4).all()

            # p(zs | zt)
            p_Xs_given_zt = torch.gather(probX_s, -1, torch.argmax(z_s.X, dim=-1, keepdim=True))
            log_p_Xs_given_zt = torch.log(p_Xs_given_zt)
            p_Es_given_zt = torch.gather(probE_s, -1, torch.argmax(z_s.E, dim=-1, keepdim=True))
            log_p_Es_given_zt = torch.log(p_Es_given_zt)

            log_p_X0 = log_p_X0 + (log_p_Xs_given_zt - log_p_Xt_given_zs)
            log_p_E0 = log_p_E0 + (log_p_Es_given_zt - log_p_Et_given_zs)

            z_s = z_t
        
        z_T = z_t
        probX_T = self.limit_dist.X.unsqueeze(0).unsqueeze(0).expand(bs, n, -1).to(z_T.X.device)
        probE_T = self.limit_dist.E.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n, n, -1).to(z_T.X.device)
        
        # p(zT)
        p_XT = torch.gather(probX_T, -1, torch.argmax(z_T.X, dim=-1, keepdim=True))
        log_p_XT = torch.log(p_XT)
        p_ET = torch.gather(probE_T, -1, torch.argmax(z_T.E, dim=-1, keepdim=True))
        log_p_ET = torch.log(p_ET)

        # log_p_X0 += log_p_XT
        # log_p_E0 += log_p_ET

        # mask
        mask_X = node_mask.unsqueeze(-1) # (bs, n, 1)
        log_p_X0[mask_X == 0] = 0
        # log_p_X0 = log_p_X0 * mask_X
        # log_p_X0[node_mask == 0] = 0 # set 'nan' = 0
        mask_E = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # (bs, 1, n) * (bs, n, 1) = (bs, n, n)
        mask_E = mask_E.unsqueeze(-1) # (bs, n, n, 1)
        log_p_E0[mask_E == 0] = 0
        # log_p_E0 = log_p_E0 * mask_E

        upper_tri_mask = torch.tril(torch.ones(n, n), diagonal=-1).unsqueeze(0).unsqueeze(-1).to(z_T.X.device) # (1, n, n, 1)
        log_p_E0 = log_p_E0 * upper_tri_mask

        log_p_X0_given_N = torch.sum(log_p_X0, dim=(1, 2))
        log_p_E0_given_N = torch.sum(log_p_E0, dim=(1, 2, 3))
        log_p_z0_given_N = log_p_X0_given_N + log_p_E0_given_N
        log_p_N =  self.node_dist.log_prob(node_mask.sum(1).long())
        log_p_z0 = log_p_z0_given_N + log_p_N

        logp_dict = {
            "G_logp": log_p_z0_given_N,
            "X_logp": log_p_X0_given_N,
            "E_logp": log_p_E0_given_N
        }

        return logp_dict

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.sampling_steps * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        if len(self.limit_dist.X.shape) == 1: # (dx_out, )
            limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        else: # (dx_out, dx_out)
            limit_X = X @ self.limit_dist.X.type_as(probX)
        
        if len(self.limit_dist.E.shape) == 1: # (de_out, )
            limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        else: # (de_out, de_out)
            limit_E = E @ self.limit_dist.E.type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.sampling_steps * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y
        #print(y0)
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probX0 = F.normalize(probX0 + 10**-30, p=1, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.model.training else 1
        t_int = torch.randint(lowest_t, self.sampling_steps + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.sampling_steps
        s_float = s_int / self.sampling_steps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = (self.test_X_logp if test else self.val_X_logp)(X * prob0.X.log()) + (self.test_E_logp if test else self.val_E_logp)(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        if self.use_full_y:
            y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        else:
            y = noisy_data['y_t'].float()
        return self.model(X, E, y, node_mask)

    def apply_noise_to_z_s(self, X, E, y, node_mask, s_int):
        """ Sample noise and apply it to the data at timestep s, where t = s + 1.
        :param X: (bs, n, dx_in)
        :param E: (bs, n, n, de_in)
        :param y: (bs, dy_in)
        :param node_mask: (bs, n)
        :param s_int: (bs, 1) - timestep s, where t = s + 1
        :return: noisy_data: dict with keys 't_int', 't', 'beta_t', 'alpha_s_bar', 'alpha_t_bar', 'X_t', 'E_t', 'y_t', 'node_mask'
        """
        t_int = s_int + 1

        t_float = t_int / self.sampling_steps
        s_float = s_int / self.sampling_steps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qt = self.transition_model.get_Qt(beta_t, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qt.X.sum(dim=2) - 1.) < 1e-4).all(), Qt.X.sum(dim=2) - 1
        assert (abs(Qt.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qt.X  # (bs, n, dx_out)
        probE = E @ Qt.E.unsqueeze(1)  # (bs, n, n, de_out)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    @torch.no_grad()
    def sample_batch(self, batch_size: int, num_nodes=None):
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
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        
        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.sampling_steps)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.sampling_steps
            t_norm = t_array / self.sampling_steps

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask,cond=True)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        
        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t,node_mask,cond=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        # TODO
        
        if (cond):
            bs, n, dxs = X_t.shape
            beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

            # Retrieve transitions matrix
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
            Qt = self.transition_model.get_Qt(beta_t, self.device)

            # Neural net predictions
            
            #CONDITIONED

            y_classes = self.output_dims['y']
            y_t = torch.tensor([1 if i == y_classes - 1 else 0 for i in range(y_classes)]).repeat(y_t.shape[0], 1).to(self.device)

            noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred_cond = self.forward(noisy_data, extra_data, node_mask)

            pred_X_cond = self.gamma * pred_cond.X
            pred_E_cond = self.gamma * pred_cond.E
            pred_y_cond = self.gamma * pred_cond.y
            
            #UNCONDITIONED
            
            y_classes = self.output_dims['y']
            y_t = torch.tensor([0 for i in range(y_classes)]).repeat(y_t.shape[0], 1).to(self.device)
            
            noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred_uncond = self.forward(noisy_data, extra_data, node_mask)
            
            pred_X_uncond = (1 - self.gamma) * pred_uncond.X
            pred_E_uncond = (1 - self.gamma) * pred_uncond.E
            pred_y_uncond = (1 - self.gamma) * pred_uncond.y
            
            pred=utils.PlaceHolder(X=pred_X_cond+pred_X_uncond,E=pred_E_cond+pred_E_uncond,y= pred_y_cond+pred_y_uncond)
            
            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
            

            p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                               Qt=Qt.X,
                                                                                               Qsb=Qsb.X,
                                                                                               Qtb=Qtb.X)

            p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                               Qt=Qt.E,
                                                                                               Qsb=Qsb.E,
                                                                                               Qtb=Qtb.E)
            # Dim of these two tensors: bs, N, d0, d_t-1
            weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
            unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

            pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
            unnormalized_prob_E = weighted_E.sum(dim=-2)
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)                      

            X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
            E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

            assert (E_s == torch.transpose(E_s, 1, 2)).all()
            assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

            out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=pred.y)
            out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=pred.y)
            
        else:
            bs, n, dxs = X_t.shape
            beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

            # Retrieve transitions matrix
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
            Qt = self.transition_model.get_Qt(beta_t, self.device)

            # Neural net predictions
            noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

            p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                               Qt=Qt.X,
                                                                                               Qsb=Qsb.X,
                                                                                               Qtb=Qtb.X)

            p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                               Qt=Qt.E,
                                                                                               Qsb=Qsb.E,
                                                                                               Qtb=Qtb.E)
            # Dim of these two tensors: bs, N, d0, d_t-1
            weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
            unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

            pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
            unnormalized_prob_E = weighted_E.sum(dim=-2)
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

            X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
            E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()


            assert (E_s == torch.transpose(E_s, 1, 2)).all()
            assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

            out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
            out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))           

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)
        
        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)
        
        t = noisy_data['t']

        extra_y = torch.cat((extra_y, t), dim=1)
        if (len(extra_y.shape) < len(noisy_data['y_t'].shape)):
            extra_y=extra_y.unsqueeze(dim=-1)
            extra_y = F.pad(input=extra_y, pad=(0, 1, 0, 1, 0, 0), mode='constant', value=0)
            
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
