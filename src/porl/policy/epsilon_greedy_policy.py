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

    # Removed commented-out print statement
    return q_values.argmax().item()


from porl.net.categorical_q_network import CategoricalQNetwork # Import for type hint

def epsilon_greedy_categorical_policy(
    state: np.ndarray,
    epsilon: float,
    q_network: CategoricalQNetwork, # Changed type hint for specificity
    action_size: int,
    device: torch.device,
) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)

    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_network.get_q_values(state_tensor)

    # Removed commented-out print statement
    return q_values.argmax().item()
