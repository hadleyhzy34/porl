import csv
import os
from datetime import datetime
from typing import Dict, Optional
import statistics
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """A logger for recording training metrics in reinforcement learning.

    Args:
        log_dir (str): Directory to store logs.
        run_name (str, optional): Name of the training run. Defaults to timestamp.
    """

    def __init__(self, log_dir: str, run_name: Optional[str] = None):
        # Create log directory with timestamp if run_name is not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Initialize CSV file for step-level logging
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["episode", "step", "reward", "loss", "epsilon"])
        self.csv_file.flush()

        # Track episode-level metrics
        self.episode_rewards = []
        self.episode_losses = []

    def log_step(
        self,
        episode: int,
        step: int,
        reward: float,
        loss: Optional[float],
        epsilon: float,
    ) -> None:
        """Log metrics for a single step.

        Args:
            episode (int): Current episode number.
            step (int): Current step number within the episode.
            reward (float): Reward for the step.
            loss (Optional[float]): Loss for the step (None if no training occurred).
            epsilon (float): Current epsilon value.
        """
        # Append to episode-level buffers
        self.episode_rewards.append(reward)
        if loss is not None:
            self.episode_losses.append(loss)

        # Write to CSV
        self.csv_writer.writerow(
            [episode, step, reward, loss if loss is not None else "", epsilon]
        )
        self.csv_file.flush()

        # Log to TensorBoard (step-level)
        global_step = episode * 1000 + step  # Assuming max 1000 steps per episode
        self.writer.add_scalar("Reward/Step", reward, global_step)
        if loss is not None:
            self.writer.add_scalar("Loss/Step", loss, global_step)
        self.writer.add_scalar("Epsilon", epsilon, global_step)

    def log_episode(self, episode: int) -> None:
        """Log summary metrics for an episode.

        Args:
            episode (int): Current episode number.
        """
        if not self.episode_rewards:
            return

        total_reward = sum(self.episode_rewards)
        avg_loss = (
            sum(self.episode_losses) / len(self.episode_losses)
            if self.episode_losses
            else None
        )

        # Log to TensorBoard (episode-level)
        self.writer.add_scalar("Reward/Episode", total_reward, episode)
        if avg_loss is not None:
            self.writer.add_scalar("Loss/Episode", avg_loss, episode)

        # # Print to console
        # print(
        #     f"Episode {episode}, Total Reward: {total_reward:.2f}, "
        #     f"Avg Loss: {avg_loss:.4f}"
        #     if avg_loss is not None
        #     else f"No loss recorded"
        # )

        # Reset episode buffers
        self.episode_rewards = []
        self.episode_losses = []

    def log_hyperparameters(self, hparams: Dict) -> None:
        """Log hyperparameters at the start of training.

        Args:
            hparams (Dict): Dictionary of hyperparameters.
        """
        self.writer.add_hparams(hparams, metric_dict={})

    def close(self) -> None:
        """Close the logger and flush all data."""
        self.csv_file.close()
        self.writer.close()
