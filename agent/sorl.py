import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
from util.util import compute_batched, update_exponential_moving_average
from agent.value_functions import TwinV
from agent.policy import GaussianPolicy, BoundedGaussianPolicy


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    # from paper: "Offline Reinforcement Learning with Implicit Q-Learning" by Ilya et al.
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class SORL(nn.Module):
    def __init__(agent,
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
        agent.device = device
        agent.backbone = backbone

        if agent.backbone is None:
            agent.v_net = TwinV(args.state_size,
                               layer_norm=args.layer_norm,
                               hidden_dim=args.hidden_dim,
                               n_hidden=args.n_hidden).to(device)

            agent.policy = BoundedGaussianPolicy(args.state_size,
                                         args.action_size,
                                         hidden_dim=args.hidden_dim,
                                         n_hidden=args.n_hidden).to(device)
        else:
            agent.backbone = agent.backbone.to(device)
            agent.v_net = TwinV(args.feature_dim,
                               layer_norm=args.layer_norm,
                               hidden_dim=args.hidden_dim,
                               n_hidden=args.n_hidden).to(device)

            agent.policy = BoundedGaussianPolicy(args.feature_dim,
                                         args.action_size,
                                         hidden_dim=args.hidden_dim,
                                         n_hidden=args.n_hidden).to(device)

        # value function
        agent.v_tgt = copy.deepcopy(agent.v_net).requires_grad_(False).to(device)
        agent.v_optimizer = torch.optim.Adam(agent.v_net.parameters(), lr=value_lr)

        # agent policy
        agent.policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=policy_lr)
        agent.lr_schedule = CosineAnnealingLR(agent.policy_optimizer, max_steps)

        agent.tau = tau
        agent.alpha = alpha
        agent.discount = discount
        agent.beta = beta

    def select_action(agent, observations):
        with torch.no_grad():
            if agent.backbone is not None:
                observations = agent.backbone(observations)
            action_distri = agent.policy(observations)
        return action_distri.mean.cpu().numpy()

    def update(agent, observations, actions, rewards, next_observations, terminals):
        # pdb.set_trace()
        # obtain latent feature
        if agent.backbone is not None:
            observations = agent.backbone(observations)
            next_observations = agent.backbone(next_observations)

        # the network will NOT update
        with torch.no_grad():
            next_v = agent.v_tgt(next_observations)  #(b,)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * agent.discount * next_v
        vs = agent.v_net.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, agent.tau) for v in vs) / len(vs)
        agent.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        agent.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(agent.v_tgt, agent.v_net, agent.beta)

        # Update policy
        # pdb.set_trace()
        v = agent.v_net(observations)
        adv = target_v - v
        weight = torch.exp(agent.alpha * adv)
        # weight = torch.exp(adv / agent.alpha)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        action_distri = agent.policy(observations.detach())
        g_loss = -action_distri.log_prob(actions)
        g_loss = torch.mean(weight * g_loss)
        agent.policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        agent.policy_optimizer.step()
        agent.lr_schedule.step()

        # # Update policy
        # v = agent.v_net(observations)
        # adv = target_v - v
        # weight = torch.exp(agent.alpha * adv)
        # # weight = torch.exp(adv / agent.alpha)
        # weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        # learned_actions = agent.policy(observations.detach())
        # g_loss = torch.mean(weight / torch.linalg.norm(learned_actions - actions, dim=-1))
        # agent.policy_optimizer.zero_grad(set_to_none=True)
        # g_loss.backward()
        # agent.policy_optimizer.step()
        # agent.lr_schedule.step()

        return v_loss.item(), g_loss.item()

    def vf_update(agent, observations, actions, rewards, next_observations, terminals):
        # pdb.set_trace()
        # obtain latent feature
        if agent.backbone is not None:
            observations = agent.backbone(observations)
            next_observations = agent.backbone(next_observations)

        # the network will NOT update
        with torch.no_grad():
            next_v = agent.v_tgt(next_observations)  #(b,)

        # Update value function
        target_v = rewards + (1. - terminals.float()) * agent.discount * next_v
        vs = agent.v_net.both(observations)
        v_loss = sum(asymmetric_l2_loss(target_v - v, agent.tau) for v in vs) / len(vs)
        agent.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        agent.v_optimizer.step()

        # Update target V network
        update_exponential_moving_average(agent.v_tgt, agent.v_net, agent.beta)

        return v_loss.item()

    def policy_update(agent, observations, actions, rewards, next_observations, terminals):
        # pdb.set_trace()
        # obtain latent feature
        if agent.backbone is not None:
            observations = agent.backbone(observations)
            next_observations = agent.backbone(next_observations)

        # pdb.set_trace()
        v = agent.v_net(observations)
        adv = target_v - v
        weight = torch.exp(agent.alpha * adv)
        # weight = torch.exp(adv / agent.alpha)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        action_distri = agent.policy(observations.detach())
        g_loss = -action_distri.log_prob(actions)
        g_loss = torch.mean(weight * g_loss)
        agent.policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        agent.policy_optimizer.step()
        agent.lr_schedule.step()

        return g_loss.item()

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
