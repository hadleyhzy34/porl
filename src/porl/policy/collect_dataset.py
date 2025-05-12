from porl.train.dqn_trainer import DQNTrainer
import pdb


def collect_dataset(env, agent: DQNTrainer, num_episodes: int = 10000) -> None:
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
            # pdb.set_trace()
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
