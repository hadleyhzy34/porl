import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
from util.util import compute_batched, update_exponential_moving_average
from agent.policy import GaussianPolicy
from agent.value_functions import TwinV

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    # from paper: "Offline Reinforcement Learning with Implicit Q-Learning" by Ilya et al.
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class POR(nn.Module):
    def __init__(agent,
                 args,
                 max_steps,
                 tau,
                 alpha,
                 backbone = None,
                 device=torch.device('cpu'),
                 value_lr=1e-4,
                 policy_lr=1e-4,
                 discount=0.99,
                 beta=0.005):
        super().__init__()
        agent.device = device
        agent.backbone = backbone
        if agent.backbone is None:
            agent.goal_policy = GaussianPolicy(args.state_size,
                                              args.state_size,
                                              hidden_dim=args.hidden_dim,
                                              n_hidden=args.n_hidden).to(agent.device)

            # state value function
            agent.vf = TwinV(args.state_size,
                            layer_norm=args.layer_norm,
                            hidden_dim=args.hidden_dim,
                            n_hidden=args.n_hidden).to(agent.device)
        else:
            agent.backbone = agent.backbone.to(device)
            agent.goal_policy = GaussianPolicy(args.feature_dim,
                                              args.state_size,
                                              hidden_dim=args.hidden_dim,
                                              n_hidden=args.n_hidden).to(agent.device)

            # state value function
            agent.vf = TwinV(args.feature_dim,
                            layer_norm=args.layer_norm,
                            hidden_dim=args.hidden_dim,
                            n_hidden=args.n_hidden).to(agent.device)

        agent.v_target = copy.deepcopy(agent.vf).requires_grad_(False).to(device)
        # agent.policy = policy.to(DEFAULT_DEVICE)
        agent.v_optimizer = torch.optim.Adam(agent.vf.parameters(), lr=value_lr)
        # agent.policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=policy_lr)
        # agent.policy_lr_schedule = CosineAnnealingLR(agent.policy_optimizer, max_steps)
        agent.goal_policy_optimizer = torch.optim.Adam(agent.goal_policy.parameters(), lr=policy_lr)
        agent.goal_lr_schedule = CosineAnnealingLR(agent.goal_policy_optimizer, max_steps)
        agent.tau = tau
        agent.alpha = alpha
        agent.discount = discount
        agent.beta = beta
        agent.step = 0
        agent.pretrain_step = 0

    def por_residual_update(agent, observations, next_observations, rewards, terminals):
        # pdb.set_trace()
        if agent.backbone is not None:
            observations = agent.backbone(observations)
            next_observations_feat = agent.backbone(next_observations)
            with torch.no_grad():
                next_v = agent.v_target(next_observations_feat)  #(b,)
        else:
            with torch.no_grad():
                next_v = agent.v_target(next_observations)  #(b,)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * agent.discount * next_v
        vs = agent.vf.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, agent.tau) for v in vs) / len(vs)
        agent.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        agent.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(agent.v_target, agent.vf, agent.beta)

        # Update goal policy
        # pdb.set_trace()
        v = agent.vf(observations.detach())
        adv = target_v - v
        # weight = torch.exp(agent.alpha * adv)
        weight = torch.exp(adv / agent.alpha)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        goal_out = agent.goal_policy(observations.detach())
        g_loss = -goal_out.log_prob(next_observations)
        if g_loss.min() <= 0:
            pdb.set_trace()
        g_loss = torch.mean(weight * g_loss)
        agent.goal_policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        agent.goal_policy_optimizer.step()
        agent.goal_lr_schedule.step()

        return v_loss.item(), g_loss.item()


        # # Update policy
        # policy_out = agent.policy(torch.concat([observations, next_observations], dim=1))
        # bc_losses = -policy_out.log_prob(actions)
        # policy_loss = torch.mean(bc_losses)
        # agent.policy_optimizer.zero_grad(set_to_none=True)
        # policy_loss.backward()
        # agent.policy_optimizer.step()
        # agent.policy_lr_schedule.step()
        #
        # # wandb
        # if (agent.step+1) % 100000 == 0:
        #     wandb.log({"v_loss": v_loss, "v_value": v.mean()}, step=agent.step)
        # agent.step += 1

    def pretrain_init(agent, b_goal_policy):
        agent.b_goal_policy = b_goal_policy.to(DEFAULT_DEVICE)
        agent.b_goal_policy_optimizer = torch.optim.Adam(agent.b_goal_policy.parameters(), lr=0.0001)

    def pretrain(agent, observations, actions, next_observations, rewards, terminals):
        # Update behavior goal policy
        b_goal_out = agent.b_goal_policy(observations)
        b_g_loss = -b_goal_out.log_prob(next_observations).mean()
        b_g_loss = torch.mean(b_g_loss)
        agent.b_goal_policy_optimizer.zero_grad(set_to_none=True)
        b_g_loss.backward()
        agent.b_goal_policy_optimizer.step()

        if (agent.pretrain_step+1) % 10000 == 0:
            wandb.log({"b_g_loss": b_g_loss}, step=agent.pretrain_step)

        agent.pretrain_step += 1

    def por_qlearning_update(agent, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            next_v = agent.v_target(next_observations)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * agent.discount * next_v
        vs = agent.vf.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, agent.tau) for v in vs) / len(vs)
        agent.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        agent.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(agent.v_target, agent.vf, agent.beta)

        # Update goal policy
        v = agent.vf(observations)
        goal_out = agent.goal_policy(observations)
        b_goal_out = agent.b_goal_policy(observations)
        g_sample = goal_out.rsample()
        g_loss1 = -agent.vf(g_sample)
        g_loss2 = -b_goal_out.log_prob(g_sample).mean()
        lmbda = agent.alpha/g_loss1.abs().mean().detach()
        g_loss = torch.mean(lmbda * g_loss1 + g_loss2)
        agent.goal_policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        agent.goal_policy_optimizer.step()
        agent.goal_lr_schedule.step()

        # Update policy
        policy_out = agent.policy(torch.concat([observations, next_observations], dim=1))
        bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(bc_losses)
        agent.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        agent.policy_optimizer.step()
        agent.policy_lr_schedule.step()

        # wandb
        if (agent.step+1) % 100000 == 0:
            wandb.log({"v_loss": v_loss, "v_value": v.mean(), "g_loss1": g_loss1.mean(), "g_loss2": g_loss2.mean()}, step=agent.step)

        agent.step += 1

    def save_pretrain(agent, filename):
        torch.save(agent.b_goal_policy.state_dict(), filename + "-behavior_goal_network")
        print(f"***save models to {filename}***")

    def load_pretrain(agent, filename):
        agent.b_goal_policy.load_state_dict(torch.load(filename + "-behavior_goal_network", map_location=DEFAULT_DEVICE))
        print(f"***load models from {filename}***")

    def save(agent, filename):
        torch.save(agent.policy.state_dict(), filename + "-policy_network")
        torch.save(agent.goal_policy.state_dict(), filename + "-goal_network")
        print(f"***save models to {filename}***")

    def load(agent, filename):
        agent.policy.load_state_dict(torch.load(filename + "-policy_network", map_location=torch.device('cpu')))
        print(f"***load the RvS policy model from {filename}***")
