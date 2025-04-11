from typing import Callable
import torch
import numpy as np
import pdb


def epsilon_greedy_policy(
    state: np.ndarray,
    epsilon: float,
    q_network: Callable[[torch.Tensor], torch.Tensor],
    action_size: int,
    device: torch.device,
) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)

    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_network(state_tensor)

    return q_values.argmax().item()
