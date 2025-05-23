import torch
from porl.env.env import lunarLander
from porl.train.c51_trainer import C51Trainer

def main():
    env, state_size, action_size = lunarLander()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = C51Trainer(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
        update_target_freq=10,
        device=device,
        atom_size=51,
        v_min=-10,
        v_max=10,
        log_dir="logs",
    )
    trainer.train_online(env)

if __name__ == "__main__":
    main()