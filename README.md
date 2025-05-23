# Autonomous Robot Navigation with Deep Reinforcement Learning

This project focuses on the development and comparison of various deep reinforcement learning algorithms for autonomous robot navigation. The Gazebo simulator provides the environment for training and evaluating these agents. Key algorithms explored in this project include Pathfinding-based Occupancy Regularization (POR), Symplectic ODE-based Reinforcement Learning (SORL), Deep Q-Networks (DQN), and Conservative Q-Learning (CQL).

## Project Structure

The repository is organized as follows:

*   `agent/`: Contains implementations of different reinforcement learning agents (e.g., POR, SORL, DQN components like policies, value functions).
*   `buffer/`: Includes code for replay buffers used in RL training.
*   `dataloader/`: Contains scripts for loading and preprocessing data, including the A* planner for expert trajectory generation.
*   `env/`: Defines the simulation environment, likely an interface with Gazebo.
*   `expert/`: Contains C++ code for the A* expert planner/controller.
*   `scripts/`: Contains various training scripts for different agents (e.g., `train_bcq.py`, `train_cql.py`, `train_dddqn.py`, `train_dqn.py`). Note: `por_train.py` and `sorl_train.py` are located in the root directory.
*   `src/`: Contains a structured/refactored version of RL components (e.g., under `porl/` - "Policy and Offline Reinforcement Learning").
*   `util/`: Contains utility functions used across the project (e.g., logging, costmap).
*   `weights/`: (Assumed) Directory for storing trained model weights.
*   `log/`: (Assumed) Directory for storing training logs and TensorBoard data.
*   `checkpoint/`: (Assumed based on `collect.py` and `preprocess.py`) Directory for storing collected raw and preprocessed datasets.

## Key Python Scripts

The following are key Python scripts located in the root directory:

*   `collect.py`: Collects experience data from the Gazebo simulation environment using multiple parallel processes.
*   `por_train.py`: Trains a Policy Order-constrained Reinforcement (POR) agent using collected/preprocessed data.
*   `preprocess.py`: Preprocesses the raw collected data. This may include generating expert trajectories using an A* planner and calculating state values.
*   `runner.py`: A script for training various RL agents (e.g., DQN, DQN_CQL) leveraging the `stable-baselines3` library.
*   `sorl_train.py`: Trains a State-Occupancy Reinforcement Learning (SORL) agent.
*   `test.py`: Evaluates the performance of a trained RL agent within the Gazebo simulation environment.

## Getting Started

This section provides instructions on how to set up and run the project.

### Prerequisites

*   **ROS (Robot Operating System):** A specific version might be required (e.g., Noetic). Ensure your ROS distribution is compatible with the Gazebo version you intend to use.
*   **Gazebo:** The simulator used for the robot environment. Version should correspond to your ROS distribution (e.g., Gazebo 11 for ROS Noetic).
*   **Python:** Python 3.8 or newer is recommended.
*   **PyTorch:** For deep learning model implementations.
*   **stable-baselines3:** For some of the RL agent training (used in `runner.py`).
*   **NumPy:** For numerical operations.
*   **Matplotlib:** Used for plotting and visualization (e.g., in `preprocess.py`, `por_train.py`).
*   **TensorBoard:** For logging and visualizing training progress.
*   **rospy:** Python client library for ROS, essential for interfacing with the ROS ecosystem.
*   Other Python dependencies are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_name>       # Replace <repository_name> with the cloned directory name
    ```

2.  **Install Python dependencies:**
    It is highly recommended to use a virtual environment (e.g., venv, conda).
    ```bash
    pip install -r requirements.txt
    ```

3.  **ROS Environment Setup:**
    *   Ensure your ROS environment is sourced: `source /opt/ros/<your_ros_distro>/setup.bash` (replace `<your_ros_distro>` with your ROS version, e.g., `noetic`).
    *   If the project includes custom ROS packages (like the `expert/` A* planner), they may need to be built using `catkin_make` or `catkin build` within a Catkin workspace. Refer to ROS documentation for setting up a workspace.

### Basic Usage / Running Scripts

The general workflow for using this project is as follows:

1.  **Collect Data:**
    Run the `collect.py` script to gather experience data from the Gazebo environment.
    ```bash
    python collect.py [arguments]
    ```
    Refer to the script's argument parser for available options.

2.  **Preprocess Data (if applicable):**
    The `preprocess.py` script is used to process the raw data collected. This might involve generating expert trajectories using the A* planner.
    ```bash
    python preprocess.py [arguments]
    ```
    Check the script for specific arguments.

3.  **Train an Agent:**
    Several scripts are available for training different agents:
    *   For POR agents:
        ```bash
        python por_train.py [arguments]
        ```
    *   For SORL agents:
        ```bash
        python sorl_train.py [arguments]
        ```
    *   For agents based on `stable-baselines3` (e.g., DQN, CQL):
        ```bash
        python runner.py [arguments]
        ```
    *   Other specific training scripts can be found in the `scripts/` directory.
    Each training script has its own set of command-line arguments for configuration.

4.  **Test/Evaluate an Agent:**
    Use `test.py` to evaluate the performance of a trained agent in the simulation.
    ```bash
    python test.py [arguments]
    ```
    Consult the script's arguments for evaluation options (e.g., loading a specific model).

**Note:** For all scripts, `[arguments]` signifies that you should pass the necessary command-line arguments. Use the `-h` or `--help` flag with any script to see its available options and required parameters.

## Usage

This section provides more details on using trained models, configuring scripts, and handling data.

### Using Trained Models

*   The training scripts (`por_train.py`, `sorl_train.py`, `runner.py`, and those in the `scripts/` directory) save the trained model weights. These are typically stored in a directory named `weights/` or a path specified via command-line arguments during training.
*   The `test.py` script is used to load these saved models and evaluate their performance in the Gazebo simulation environment. You'll need to provide the path to the specific model file you want to test.
*   **Example:**
    ```bash
    python test.py --model_path weights/your_trained_model.pt [other_arguments]
    ```
    Please replace `weights/your_trained_model.pt` with the actual path to your model and `[other_arguments]` with any other necessary parameters for `test.py`. Refer to `python test.py --help` for the exact argument names and options.

### Configuration

*   As mentioned in the "Basic Usage" section, most Python scripts in this project (e.g., `collect.py`, `por_train.py`, `sorl_train.py`, `runner.py`, `test.py`, and scripts in `scripts/`) use Python's `argparse` module for command-line argument parsing.
*   To discover all configurable parameters for a script, run it with the `-h` or `--help` flag. For example:
    ```bash
    python por_train.py --help
    python runner.py --help
    ```
*   Key configurable aspects often include:
    *   Hyperparameters for training (e.g., learning rate, batch size, discount factor).
    *   Network architecture details (e.g., number of layers, hidden units).
    *   Training duration (e.g., number of episodes, timesteps).
    *   Paths for saving models, logs, and datasets.
    *   Environment-specific parameters (e.g., Gazebo world name, robot namespace, state and action dimensions).
    *   Algorithm-specific settings.

### Data

*   Raw data collected by `collect.py` is typically saved in the `checkpoint/` directory, often as PyTorch tensor files (e.g., `checkpoint/dataset_raw.pt`, `checkpoint/dataset_expert.pt`).
*   This data usually consists of sequences of experiences, where each experience might be a dictionary or tuple containing information like:
    *   `state`: The robot's state (e.g., laser scan, odometry).
    *   `action`: The action taken by the robot.
    *   `reward`: The reward received.
    *   `next_state`: The subsequent state.
    *   `done`: A flag indicating if the episode terminated.
*   The `preprocess.py` script takes this raw data, potentially processes it (e.g., by aligning with expert trajectories from A*), and saves it, often in a similar format, for use by the training scripts. The training scripts then load this (preprocessed) data for model training.

## Contributing

We welcome contributions to enhance and expand this project! If you're interested in contributing, here are some ways you can help:

*   **Reporting Bugs:** If you encounter any bugs or unexpected behavior, please open an issue on the project's issue tracker.
*   **Proposing New Features:** Have an idea for a new algorithm, environment, or utility? Feel free to suggest it by opening an issue.
*   **Improving Documentation:** Enhancements to the README or other documentation are always welcome.
*   **Submitting Pull Requests:** For new algorithms, bug fixes, or other improvements, please follow this general workflow:
    1.  Fork the repository.
    2.  Create a new branch for your feature or fix (e.g., `feature/my-new-algorithm` or `fix/some-bug`).
    3.  Make your changes and commit them with clear messages.
    4.  Open a pull request against the main branch of this repository.

Please check the GitHub Issues page for current tasks, discussions, and areas where you can contribute.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
