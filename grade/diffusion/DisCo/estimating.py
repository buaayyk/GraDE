import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

from utils import *
from auxiliary_features import *

class Estimator():
    def __init__(self, n_node_type, n_edge_type, num_steps=1000, min_t = 0.01, 
                        add_auxiliary_feature=None, device='cpu', BAR=True, level_fea=0, extra_time=False):
        assert add_auxiliary_feature != None
        self.n_node_type = n_node_type
        self.n_edge_type = n_edge_type
        self.num_steps = num_steps
        self.min_t = min_t
        self.add_auxiliary_feature = add_auxiliary_feature
        self.device = device
        self.level_fea = level_fea
        self.extra_time = extra_time

        if n_node_type <= 1:
            self.diffuse_node = False # whether to diffuse node or not, but always diffuse edges
        else:
            self.diffuse_node = True
        
        self.BAR = BAR

    def logp(self, diffuser, model, data):
        # N is the number of samples
        N = data.batch_size
        t = 1.0
        n_node_type = self.n_node_type
        n_edge_type = self.n_edge_type
        min_t = self.min_t
        num_steps = self.num_steps
        eps_ratio = 1e-9 # from Cambell's code
        device = self.device
        data = data.to(device)
        
        with torch.no_grad():
            # --------------- Initializing E, X, and time t --------------------
            X_0, E_0, node_mask, level = to_dense(data)
            edge_mask = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2))
            n_node = X_0.shape[1]
            ts = torch.tensor(np.concatenate((np.linspace(t, min_t, num_steps), np.array([0]))), device=device).flip(0).unsqueeze(1).expand(-1, N).float()

            X_s, E_s = X_0, E_0
            pbar = tqdm(enumerate(ts[0:-1])) if self.BAR else enumerate(ts[0:-1])
            X_logp = torch.zeros((N, n_node), device=device).float() # \log{p(X_0)}
            E_logp = torch.zeros((N, n_node, n_node), device=device).float() # \log{p(E_0)}
            for idx, t in pbar:
                s = ts[idx]
                t = ts[idx+1]
                h = (t - s) # (N,) tau
                
                # --------------- Forward sample from s to t ---------------
                X_t_idx, E_t_idx = diffuser.forward_diffusion_from_s_to_t(X_s, E_s, s, t)

                # --------------- Forward probability -----------
                E_qts, X_qts = diffuser.transition_from_s_to_t(s, t)
                X_s_idx = torch.max(X_s, dim=2)[1]
                E_s_idx = torch.max(E_s, dim=3)[1]
                forward_X_probs = X_qts[
                    torch.arange(N, device=device).repeat_interleave(n_node), 
                    X_s_idx.flatten().long(),
                    X_t_idx.flatten().long()
                ].view(N, n_node)
                forward_E_probs = E_qts[
                    torch.arange(N, device=device).repeat_interleave(n_node * n_node), 
                    E_s_idx.flatten().long(),
                    E_t_idx.flatten().long()
                ].view(N, n_node, n_node)
                forward_X_logp = torch.log(forward_X_probs) * node_mask # (N, n_node), \log{q_{t|s}(X_t | G_s)}
                forward_E_logp = torch.log(forward_E_probs) * edge_mask # (N, n_node, n_node), \log{q_{t|s}(E_t | G_s)}

                # --------------- Reverse probability -----------
                X_t = F.one_hot(X_t_idx, num_classes=n_node_type).float()
                E_t = F.one_hot(E_t_idx, num_classes=n_edge_type).float()
                X_t_with_auxiliary, E_t_with_auxiliary, y_t_with_auxiliary = self.add_auxiliary_feature(X_t, E_t, node_mask)
                if self.extra_time:
                    y_t_with_auxiliary = torch.cat([y_t_with_auxiliary, t.unsqueeze(-1)], dim=-1) # TODO
                X_p0t, E_p0t = model(X_t_with_auxiliary, E_t_with_auxiliary, y_t_with_auxiliary, node_mask, level, self.level_fea)
                E_qt0, X_qt0 = diffuser.transition(t)
                E_rate_t, X_rate_t = diffuser.rate(t)
                
                # Reverse edge probability
                
                E_p0t = E_p0t.view(N, n_node * n_node, n_edge_type) # (N, n_node * n_node, n_edge_type)
                E_p0t = F.softmax(E_p0t, dim=-1)

                E_qt0_denom = E_qt0[
                    torch.arange(N, device=device).repeat_interleave(n_node * n_node * n_edge_type), 
                    torch.arange(n_edge_type, device=device).repeat(N * n_node * n_node), 
                    E_t_idx.long().flatten().repeat_interleave(n_edge_type)
                ].view(N, n_node * n_node, n_edge_type) + eps_ratio
                
                E_qt0_numer = E_qt0
                
                forward_E_rates = E_rate_t[
                    torch.arange(N, device=device).repeat_interleave(n_node * n_node * n_edge_type),
                    torch.arange(n_edge_type, device=device).repeat(N * n_node * n_node),
                    E_t_idx.long().flatten().repeat_interleave(n_edge_type)
                ].view(N, n_node * n_node, n_edge_type)

                E_inner_sum = (E_p0t / E_qt0_denom) @ E_qt0_numer # (N, n_node * n_node, n_edge_type)

                reverse_E_rates = forward_E_rates * E_inner_sum # (N, n_node * n_node, n_edge_type)

                current_E_state_mask = F.one_hot(E_t_idx.view(N, n_node * n_node), num_classes=n_edge_type).bool()
                off_diagonal_E_rates = torch.where(current_E_state_mask, torch.zeros_like(reverse_E_rates), torch.clamp(reverse_E_rates, min=0.0))
                total_outflow_E_rates = off_diagonal_E_rates.sum(dim=-1)
                diagonal_E_rates = (-total_outflow_E_rates).unsqueeze(-1)
                reverse_E_rates = off_diagonal_E_rates + diagonal_E_rates * current_E_state_mask.float()

                # reverse_E_rates_from_t_to_s = reverse_E_rates[
                #     torch.arange(N).repeat_interleave(n_node * n_node),
                #     torch.arange(n_node * n_node).repeat(N),
                #     E_s_idx.long().flatten()
                # ].view(N, n_node, n_node)

                # reverse_E_probs = h.unsqueeze(1).unsqueeze(2) * reverse_E_rates_from_t_to_s + (E_s_idx == E_t_idx).float()

                reverse_E_rates_no_jump = reverse_E_rates[
                    torch.arange(N).repeat_interleave(n_node * n_node),
                    torch.arange(n_node * n_node).repeat(N),
                    E_t_idx.long().flatten()
                ].view(N, n_node, n_node)

                reverse_E_rates_jump = reverse_E_rates[
                    torch.arange(N).repeat_interleave(n_node * n_node),
                    torch.arange(n_node * n_node).repeat(N),
                    E_s_idx.long().flatten()
                ].view(N, n_node, n_node)

                reverse_E_jump_probs = (1 - torch.exp(h.unsqueeze(1).unsqueeze(2) * reverse_E_rates_no_jump)) * reverse_E_rates_jump / (-reverse_E_rates_no_jump)
                reverse_E_no_jump_probs = torch.exp(h.unsqueeze(1).unsqueeze(2) * reverse_E_rates_no_jump)

                reverse_E_probs = reverse_E_jump_probs * (E_s_idx != E_t_idx) + reverse_E_no_jump_probs * (E_s_idx == E_t_idx)

                reverse_E_logp = torch.log(reverse_E_probs) * edge_mask # (N, n_node, n_node), \log{p_{s|t}^{\theta}(E_s | G_t)}

                # Reverse node probability
                if self.diffuse_node:
                    X_p0t = F.softmax(X_p0t, dim=-1)

                    X_qt0_denom = X_qt0[
                        torch.arange(N, device=device).repeat_interleave(n_node * n_node_type),
                        torch.arange(n_node_type, device=device).repeat(N * n_node), 
                        X_t_idx.long().flatten().repeat_interleave(n_node_type)
                    ].view(N, n_node, n_node_type) + eps_ratio

                    X_qt0_numer = X_qt0

                    forward_X_rates = X_rate_t[
                        torch.arange(N, device=device).repeat_interleave(n_node * n_node_type),
                        torch.arange(n_node_type, device=device).repeat(N * n_node),
                        X_t_idx.long().flatten().repeat_interleave(n_node_type)
                    ].view(N, n_node, n_node_type)

                    X_inner_sum = (X_p0t / X_qt0_denom) @ X_qt0_numer # (N, n_node, n_node_type)

                    reverse_X_rates = forward_X_rates * X_inner_sum # (N, n_node, n_node_type)

                    current_X_state_mask = F.one_hot(X_t_idx.view(N, n_node), num_classes=n_node_type).bool()
                    off_diagonal_X_rates = torch.where(current_X_state_mask, torch.zeros_like(reverse_X_rates), torch.clamp(reverse_X_rates, min=0.0))
                    total_outflow_X_rates = off_diagonal_X_rates.sum(dim=-1)
                    diagonal_X_rates = (-total_outflow_X_rates).unsqueeze(-1)
                    reverse_X_rates = off_diagonal_X_rates + diagonal_X_rates * current_X_state_mask.float()

                    # reverse_X_rates_from_t_to_s = reverse_X_rates[
                    #     torch.arange(N).repeat_interleave(n_node),
                    #     torch.arange(n_node).repeat(N),
                    #     X_s_idx.long().flatten()
                    # ].view(N, n_node)

                    # reverse_X_probs = h.unsqueeze(1) * reverse_X_rates_from_t_to_s + (X_s_idx == X_t_idx).float()

                    reverse_X_rates_no_jump = reverse_X_rates[
                        torch.arange(N).repeat_interleave(n_node), 
                        torch.arange(n_node).repeat(N),
                        X_t_idx.long().flatten()
                    ].view(N, n_node)

                    reverse_X_rates_jump = reverse_X_rates[
                        torch.arange(N).repeat_interleave(n_node),
                        torch.arange(n_node).repeat(N),
                        X_s_idx.long().flatten()
                    ].view(N, n_node)

                    reverse_X_jump_probs = (1 - torch.exp(h.unsqueeze(1) * reverse_X_rates_no_jump)) * reverse_X_rates_jump / (-reverse_X_rates_no_jump)
                    reverse_X_no_jump_probs = torch.exp(h.unsqueeze(1) * reverse_X_rates_no_jump)

                    reverse_X_probs = reverse_X_jump_probs * (X_s_idx != X_t_idx) + reverse_X_no_jump_probs * (X_s_idx == X_t_idx)

                    reverse_X_logp = torch.log(reverse_X_probs) * node_mask # (N, n_node), \log{p_{s|t}^{\theta}(X_s|G_t)} 
                else:
                    reverse_X_logp = torch.zeros((N, n_node), device=device) # (N, n_node), \log{p_{s|t}^{\theta}(X_s|G_t)}

                # ----------- Update log p -----------
                X_logp += (reverse_X_logp - forward_X_logp)
                E_logp += (reverse_E_logp - forward_E_logp)
                X_s, E_s = X_t, E_t

            # ----------- Reference distribution probability log p ------------
            edge_reference_distribution, node_reference_distribution = diffuser.get_reference_distribution(N) # (N, edge_S), (N, node_S)
            ref_X_probs = node_reference_distribution[
                torch.arange(N).repeat_interleave(n_node),
                X_t_idx.long().flatten()
            ].view(N, n_node)
            ref_E_probs = edge_reference_distribution[
                torch.arange(N).repeat_interleave(n_node * n_node),
                E_t_idx.long().flatten() 
            ].view(N, n_node, n_node)
            ref_X_logp = torch.log(ref_X_probs) * node_mask # (N, n_node), \log{\pi_{ref}(X_t)}
            ref_E_logp = torch.log(ref_E_probs) * edge_mask # (N, n_node), \log{\pi_{ref}(E_t)}
            # TODO: add \pi_{ref} or not?
            # X_logp += ref_X_logp
            # E_logp += ref_E_logp
            X_logp = torch.sum(X_logp * node_mask, dim=1)
            E_logp = torch.sum((E_logp * edge_mask).triu(diagonal=1), dim=[1, 2])
            G_logp = X_logp + E_logp
            
            logp_dict = {
                "G_logp": G_logp,
                "E_logp": E_logp,
                "X_logp": X_logp
            }

            return logp_dict