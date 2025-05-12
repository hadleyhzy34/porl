import torch
import numpy as np
import pdb
import torch.nn.functional as F
from porl.train.bcq_trainer import BCQTrainer


def collect_dataset(env, agent: BCQTrainer, num_episodes: int = 1000) -> None:
    """Collect an offline dataset using a random policy."""
    # dataset = []
    print(f"start collecting dataset: epochs {num_episodes}\n")
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random policy
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state


def bcq_behavior_pretrain(agent: BCQTrainer) -> None:
    """Pre-train the behavior policy model on the offline dataset.
    Args:
        agent ([TODO:parameter]): [TODO:description]

    Returns:
        [TODO:return]
    """
    # pdb.set_trace()
    print(f"start pretraining behavior policy: epochs {agent.num_epochs}\n")
    for epoch in range(agent.num_epochs):
        states, actions, _, _, _ = agent.replay_buffer.sample(agent.batch_size)
        probs = agent.behavior_policy(states)  # Shape: (batch_size, action_size)
        # log_probs = F.log_softmax(probs, dim=-1)
        # loss = F.nll_loss(log_probs, actions)

        logits = agent.behavior_policy.network(states)  # Shape: (batch_size, action_size)
        loss = F.cross_entropy(logits, actions)

        agent.behavior_optimizer.zero_grad()
        loss.backward()
        agent.behavior_optimizer.step()

        if epoch % 10 == 0:
            print(f"Behavior Policy Epoch {epoch}, Loss: {loss.item():.4f}")


def bcq_learn(agent: BCQTrainer) -> float:
    states, actions, rewards, next_states, dones = agent.replay_buffer.sample(
        agent.batch_size
    )

    # Compute allowed actions for next states
    with torch.no_grad():
        # pdb.set_trace()
        action_mask = agent.behavior_policy.sample(
            next_states, agent.threshold
        )  # Shape: (batch_size, action_size)

        next_q_values = agent.target_network(
            next_states
        )  # Shape: (batch_size, action_size)

        # Mask out actions with low probability
        masked_q_values = (
            next_q_values + (action_mask - 1) * 1e10
        )  # Large negative penalty for masked actions

        next_actions = masked_q_values.argmax(dim=1, keepdim=True)
        target_q_values = next_q_values.gather(1, next_actions).squeeze(1)
        targets = rewards + agent.gamma * target_q_values * (1 - dones)

    # Compute Q-values for current state-action pairs
    q_values = agent.q_network(states)
    # q_values = agent.behavior_policy(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(q_values, targets)

    # Update Q-network
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()
