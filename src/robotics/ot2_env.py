import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from sim_class import Simulation


class OT2ENV(gym.Env):
    """
    Gym environment wrapper for the PyBullet OT-2 Robot simulation.

    Observation Space:
        - Joint positions (3 per agent)
        - Joint velocities (3 per agent)
        - Pipette position (3 per agent)

    Action Space:
        - Continuous control for x, y, z velocities
        - Discrete control for drop action (0 or 1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(
        self,
        num_agents: int = 1,
        render_mode: str | None = None,
        target: np.ndarray = np.array([0.0, 0.0, 0.1]),
        max_steps: int = 1000,
    ) -> None:
        super().__init__()

        self.num_agents = num_agents
        self.render_mode = render_mode
        self.target = np.array(target)
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize the simulation
        render = render_mode == "human"
        rgb_array = render_mode == "rgb_array"
        self.sim = Simulation(num_agents=num_agents, render=render, rgb_array=rgb_array)

        # Define action space for each agent: [x_velocity, y_velocity, z_velocity, drop]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0] * num_agents),
            high=np.array([1.0, 1.0, 1.0, 1.0] * num_agents),
            dtype=np.float64,
        )

        # Define observation space
        obs_dim = 9 * num_agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Workspace limits (with small margin for safety)
        self.workspace_limits = {
            "x": (-0.187, 0.253),
            "y": (-0.1705, 0.2195),
            "z": (0.1194, 0.2896),
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset step counter
        self.current_step = 0

        # Reset the simulation
        states = self.sim.reset(num_agents=self.num_agents)

        obs = self._get_obs(states)
        info = self._get_info(states)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Reshape action for multiple agents
        actions = self._process_action(action)

        # Run simulation
        states = self.sim.run(actions, num_steps=1)

        # Get observation
        obs = self._get_obs(states)

        # Calculate reward
        reward = self._compute_reward(states, actions)

        # Check termination conditions
        terminated = self._is_terminated(states)

        # Check if target reached (distance <= threshold)
        distance = self._get_distance_to_target(states)

        truncated = self.current_step >= self.max_steps

        # Additional info
        info = self._get_info(states)
        info["distance_to_target"] = distance

        return obs, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert flat action array to list of actions per agent."""
        action = np.array(action, dtype=np.float64)
        actions = []

        for i in range(self.num_agents):
            start_idx = i * 4
            agent_action = action[start_idx : start_idx + 4]

            # Clip velocities
            agent_action[:3] = np.clip(agent_action[:3], -1.0, 1.0)

            # Threshold drop action (>0.5 means drop)
            agent_action[3] = 1.0 if agent_action[3] > 0.5 else 0.0

            actions.append(agent_action.tolist())

        return np.array(actions)

    def _get_obs(self, states: dict) -> np.ndarray:
        """Convert simulation states to observation array."""
        obs = []

        for robot_key in sorted(states.keys()):
            robot_state = states[robot_key]

            # Joint positions
            for i in range(3):
                obs.append(robot_state["joint_states"][f"joint_{i}"]["position"])

            # Joint velocities
            for i in range(3):
                obs.append(robot_state["joint_states"][f"joint_{i}"]["velocity"])

            # Pipette position
            obs.extend(robot_state["pipette_position"])

        return np.array(obs, dtype=np.float64)

    def _get_info(self, states: dict) -> dict:
        """Get additional information about the environment state."""
        info = {
            "step": self.current_step,
            "num_droplets": len(self.sim.sphereIds),
            "droplet_positions": self.sim.droplet_positions.copy(),
            "pipette_positions": {},
        }

        for robot_key in states.keys():
            info["pipette_positions"][robot_key] = states[robot_key]["pipette_position"]

        return info

    def _get_distance_to_target(self, states: dict) -> float:
        """Calculate distance from pipette to target (for first agent)."""
        robot_key = sorted(states.keys())[0]
        pipette_pos = np.array(states[robot_key]["pipette_position"])
        return np.linalg.norm(pipette_pos - self.target)

    def _compute_reward(self, states: dict, actions: np.ndarray) -> float:
        """Compute reward based on the current state and actions."""
        reward = 0.0

        for i, robot_key in enumerate(sorted(states.keys())):
            pipette_pos = np.array(states[robot_key]["pipette_position"])

            # Penalty for going out of bounds
            if not self._is_in_workspace(pipette_pos):
                reward -= 100.0

            # Small step penalty to encourage efficiency
            reward -= 0.01

            # Distance-based reward (negative distance)
            distance = np.linalg.norm(pipette_pos - self.target)
            reward -= distance

        return reward

    def _is_terminated(self, states: dict) -> bool:
        """Check if episode should terminate."""
        # for robot_key in states.keys():
        #     pipette_pos = states[robot_key]["pipette_position"]
        #     if not self._is_in_workspace(pipette_pos):
        #         return True
        return False

    def _is_in_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within workspace limits."""
        x, y, z = position
        return (
            self.workspace_limits["x"][0] <= x <= self.workspace_limits["x"][1]
            and self.workspace_limits["y"][0] <= y <= self.workspace_limits["y"][1]
            and self.workspace_limits["z"][0] <= z <= self.workspace_limits["z"][1]
        )

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            if hasattr(self.sim, "current_frame"):
                return self.sim.current_frame
            return None
        return None

    def close(self):
        """Close the environment and cleanup."""
        self.sim.close()


register(
    id="OT2ENV-v0",
    entry_point="ot2_env:OT2ENV",
    kwargs={
        "num_agents": 1,
        "render_mode": "human",
        "target": [0.0, 0.0, 0.1],
        "max_steps": 1000,
    },
)
