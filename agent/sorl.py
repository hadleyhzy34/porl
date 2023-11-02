import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
from util.util import compute_batched, update_exponential_moving_average
from agent.value_functions import TwinV
from agent.policy import GaussianPolicy


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    # from paper: "Offline Reinforcement Learning with Implicit Q-Learning" by Ilya et al.
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class SORL(nn.Module):
    def __init__(self,
                 args,
                 max_steps,
                 tau,
                 alpha,
                 device=torch.device('cpu'),
                 backbone = None,
                 value_lr=1e-4,
                 policy_lr=1e-4,
                 discount=0.99,
                 beta=0.005):
        super().__init__()
        self.device = device
        self.backbone = backbone

        if self.backbone is None:
            self.v_net = TwinV(args.state_size,
                               layer_norm=args.layer_norm,
                               hidden_dim=args.hidden_dim,
                               n_hidden=args.n_hidden).to(device)

            self.policy = GaussianPolicy(args.state_size,
                                         args.action_size,
                                         hidden_dim=args.hidden_dim,
                                         n_hidden=args.n_hidden).to(device)
        else:
            self.backbone = self.backbone.to(device)
            self.v_net = TwinV(args.feature_dim,
                               layer_norm=args.layer_norm,
                               hidden_dim=args.hidden_dim,
                               n_hidden=args.n_hidden).to(device)

            self.policy = GaussianPolicy(args.feature_dim,
                                         args.action_size,
                                         hidden_dim=args.hidden_dim,
                                         n_hidden=args.n_hidden).to(device)

        # value function
        self.v_tgt = copy.deepcopy(self.v_net).requires_grad_(False).to(device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=value_lr)

        # agent policy
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

        self.tau = tau
        self.alpha = alpha
        self.discount = discount
        self.beta = beta

    def select_action(self, observations):
        action_distri = self.policy(observations)
        return action_distri.mean.cpu().numpy()

    def update(self, observations, actions, rewards, next_observations, terminals):
        # pdb.set_trace()
        # obtain latent feature
        if self.backbone is not None:
            observations = self.backbone(observations)
            next_observations = self.backbone(observations)

        # the network will NOT update
        with torch.no_grad():
            next_v = self.v_tgt(next_observations)  #(b,)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * self.discount * next_v
        vs = self.v_net.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, self.tau) for v in vs) / len(vs)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(self.v_tgt, self.v_net, self.beta)

        # Update policy
        v = self.v_net(observations)
        adv = target_v - v
        weight = torch.exp(self.alpha * adv)
        # weight = torch.exp(adv / self.alpha)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        action_distri = self.policy(observations.detach())
        g_loss = -action_distri.log_prob(actions)
        g_loss = torch.mean(weight * g_loss)
        self.policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        self.policy_optimizer.step()
        self.lr_schedule.step()

        return v_loss.item(), g_loss.item()

    def pretrain_init(self, b_goal_policy):
        self.b_goal_policy = b_goal_policy.to(DEFAULT_DEVICE)
        self.b_goal_policy_optimizer = torch.optim.Adam(self.b_goal_policy.parameters(), lr=0.0001)

    def pretrain(self, observations, actions, next_observations, rewards, terminals):
        # Update behavior goal policy
        b_goal_out = self.b_goal_policy(observations)
        b_g_loss = -b_goal_out.log_prob(next_observations).mean()
        b_g_loss = torch.mean(b_g_loss)
        self.b_goal_policy_optimizer.zero_grad(set_to_none=True)
        b_g_loss.backward()
        self.b_goal_policy_optimizer.step()

        if (self.pretrain_step+1) % 10000 == 0:
            wandb.log({"b_g_loss": b_g_loss}, step=self.pretrain_step)

        self.pretrain_step += 1

    def save_pretrain(self, filename):
        torch.save(self.b_goal_policy.state_dict(), filename + "-behavior_goal_network")
        print(f"***save models to {filename}***")

    def load_pretrain(self, filename):
        self.b_goal_policy.load_state_dict(torch.load(filename + "-behavior_goal_network", map_location=DEFAULT_DEVICE))
        print(f"***load models from {filename}***")

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "-policy_network")
        torch.save(self.goal_policy.state_dict(), filename + "-goal_network")
        print(f"***save models to {filename}***")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "-policy_network", map_location=torch.device('cpu')))
        print(f"***load the RvS policy model from {filename}***")
