import torch
import numpy as np
import pdb
import torch.nn.functional as F


def dqn_learn(agent) -> float:
    # Sample directly as tensors
    states, actions, rewards, next_states, dones = agent.replay_buffer.sample(
        agent.batch_size
    )

    with torch.no_grad():
        next_q_values = agent.target_network(next_states)
        max_next_q_values = next_q_values.max(dim=1)[0]
        targets = rewards + agent.gamma * max_next_q_values * (1 - dones)

    q_values = agent.q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(q_values, targets)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()


def ddqn_learn(agent) -> float:
    # Sample directly as tensors
    states, actions, rewards, next_states, dones = agent.replay_buffer.sample(
        agent.batch_size
    )

    with torch.no_grad():
        next_actions = agent.q_network(next_states).argmax(dim=1, keepdim=True)
        max_next_q_values = (
            agent.target_network(next_states).gather(1, next_actions).squeeze(1)
        )
        targets = rewards + agent.gamma * max_next_q_values * (1 - dones)

    q_values = agent.q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(q_values, targets)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()
