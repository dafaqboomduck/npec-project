import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from .sim_class import Simulation


class OT2ENV(gym.Env):
    """
    Gym environment wrapper for the PyBullet OT-2 Robot simulation.

    Observation Space:
        - Joint positions (3 per agent)
        - Relative Vector (3 per agent)
        - Pipette position (3 per agent)
        - Target

    Action Space:
        - Continuous control for x, y, z velocities
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 240}

    def __init__(
        self,
        num_agents: int = 1,
        render_mode: str | None = None,
        target: np.ndarray | None = None,
        target_threshold: float = 0.001,
        dwell_steps: int = 10,  # Steps to stay at target (10-20 range)
        max_steps: int = 1000,
    ) -> None:
        super().__init__()

        self.num_agents = num_agents
        self.render_mode = render_mode
        self.target = np.array(target) if target else None
        self.target_threshold = target_threshold
        self.dwell_steps = dwell_steps
        self.max_steps = max_steps
        self.current_step = 0
        self.current_reward = 0
        self._last_distance = None
        self._steps_at_target = {i: 0 for i in range(self.num_agents)}
        self._last_distance = {i: None for i in range(self.num_agents)}
        self._milestones_reached = {
            i: {"10mm": False, "5mm": False, "2mm": False, "1mm": False}
            for i in range(self.num_agents)
        }

        # Initialize the simulation
        render = render_mode == "human"
        rgb_array = render_mode == "rgb_array"
        self.sim = Simulation(num_agents=num_agents, render=render, rgb_array=rgb_array)

        # Define action space for each agent: [x_velocity, y_velocity, z_velocity, drop]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0] * num_agents),
            high=np.array([1.0, 1.0, 1.0] * num_agents),
            dtype=np.float64,
        )

        # Define observation space
        obs_dim = (9 * num_agents) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Workspace limits (with small margin for safety)
        self.workspace_limits = {
            "x": (-0.1875, 0.2532),
            "y": (-0.17010, 0.2198),
            "z": (0.1195, 0.2903),
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Allow target to be passed via options
        if options is not None and "target" in options:
            self.target = options["target"]
        else:
            self.target = self._choose_random_target()

        # Reset step counter
        self.current_step = 0  # <-- THIS WAS MISSING

        # Reset reward tracking
        self._steps_at_target = {i: 0 for i in range(self.num_agents)}
        self._last_distance = {i: None for i in range(self.num_agents)}
        self._milestones_reached = {
            i: {"10mm": False, "5mm": False, "2mm": False, "1mm": False}
            for i in range(self.num_agents)
        }

        # Reset the simulation
        _ = self.sim.reset(num_agents=self.num_agents)

        random_start = self._choose_random_target()

        # Set pipette to a random starting position within the workspace
        self.sim.set_start_position(*random_start)
        states = self.sim.get_states()

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
        reward = self._compute_reward(states)

        # Check if at target and update dwell counter
        distance = self._get_distance_to_target(states)

        # Check termination conditions
        terminated = self._is_terminated(states)

        truncated = self.current_step >= self.max_steps

        # Additional info
        info = self._get_info(states)
        info["distance_to_target"] = distance
        info["steps_at_target"] = self._steps_at_target[0]
        info["dwell_progress"] = f"{self._steps_at_target[0]}/{self.dwell_steps}"

        return obs, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert flat action array to list of actions per agent."""
        action = np.array(action, dtype=np.float64)
        actions = []

        for i in range(self.num_agents):
            start_idx = i * 3
            agent_action = action[start_idx : start_idx + 3]

            # Clip velocities
            agent_action = np.clip(agent_action, -1.0, 1.0)

            agent_action = agent_action.tolist()
            agent_action.append(0)
            actions.append(agent_action)

        return np.array(actions)

    def _get_obs(self, states: dict) -> np.ndarray:
        """Convert simulation states to observation array."""
        obs = []

        # Normalize Target
        norm_target = self._normalize_position(self.target)

        for robot_key in sorted(states.keys()):
            robot_state = states[robot_key]

            # 1. Normalize Pipette Position
            pipette_pos = np.array(robot_state["pipette_position"])
            norm_pipette = self._normalize_position(pipette_pos)

            # 2. Relative Vector(Target - Pipette)
            relative_vector = norm_target - norm_pipette

            # 3. Joint velocities (Already -1 to 1, no scaling needed)
            velocities = [
                robot_state["joint_states"][f"joint_{i}"]["velocity"] for i in range(3)
            ]
            # Assemble: 3 (vel) + 3 (pipette) + 3 (rel_vector) = 9 per agent
            obs.extend(velocities)
            obs.extend(norm_pipette)
            obs.extend(relative_vector)

        # Target position: + 3
        obs.extend(norm_target)

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

    def _choose_random_target(self) -> np.ndarray:
        random_point = []
        margin = 0.005  # 5mm safety margin from the physical limits
        for _, limits in self.workspace_limits.items():
            random_point.append(
                np.random.uniform(limits[0] + margin, limits[1] - margin)
            )
        return np.array(random_point)

    def _get_distance_to_target(self, states: dict) -> float:
        """Calculate distance from pipette to target (for first agent)."""
        robot_key = sorted(states.keys())[0]
        pipette_pos = np.array(states[robot_key]["pipette_position"])
        return np.linalg.norm(pipette_pos - self.target)

    def _compute_reward(self, states: dict) -> float:
        """Fixed reward function - milestones only fire once."""
        reward = 0.0

        for i, robot_key in enumerate(sorted(states.keys())):
            pipette_pos = np.array(states[robot_key]["pipette_position"])
            dist = np.linalg.norm(pipette_pos - self.target)

            velocities = [
                states[robot_key]["joint_states"][f"joint_{j}"]["velocity"]
                for j in range(3)
            ]
            velocity_magnitude = np.linalg.norm(velocities)

            # 1: Distance Improvement (Dense Shaping)
            if self._last_distance[i] is not None:
                delta = self._last_distance[i] - dist
                distance_reward = np.clip(delta * 100, -1.0, 1.0)
                reward += distance_reward

            self._last_distance[i] = dist

            # 2: Proximity Bonus (Exponential Well)
            precision_reward = np.exp(-dist / 0.01) * 0.2
            reward += precision_reward

            # 3: Velocity Penalty Near Target
            proximity_factor = max(0, 1 - dist / 0.02)  # Active within 2cm
            velocity_penalty = proximity_factor * velocity_magnitude * 0.3
            reward -= velocity_penalty

            # 4: Dwell Reward (Stability)
            if dist <= self.target_threshold:
                self._steps_at_target[i] += 1

                # Base reward for being at target
                dwell_base = 0.3

                # Progressive bonus (diminishing returns)
                # Steps 1-10 give increasing bonus, then caps
                progress = min(self._steps_at_target[i] / self.dwell_steps, 1.0)
                dwell_progress = progress * 0.7

                reward += dwell_base + dwell_progress
            else:
                # Reset counter if we leave the target zone
                if self._steps_at_target[i] > 0:
                    reward -= 0.5  # Penalty for leaving target zone
                self._steps_at_target[i] = 0

            # 5: Milestone Bonuses (Sparse, One-Time)
            milestones = self._milestones_reached[i]

            if dist < 0.010 and not milestones["10mm"]:  # Within 1cm
                reward += 1.0
                milestones["10mm"] = True

            if dist < 0.005 and not milestones["5mm"]:  # Within 5mm
                reward += 1.5
                milestones["5mm"] = True

            if dist < 0.002 and not milestones["2mm"]:  # Within 2mm
                reward += 2.5
                milestones["2mm"] = True

            if dist < 0.001 and not milestones["1mm"]:  # Within 1mm
                reward += 4.0
                milestones["1mm"] = True

            # 6: Success Bonus (Terminal)
            if self._steps_at_target[i] >= self.dwell_steps:
                reward += 20.0

            # 7: Boundary Penalty (Safety)
            if not self._is_in_workspace(pipette_pos):
                overshoot = self._calculate_overshoot(pipette_pos)
                boundary_penalty = 3.0 + overshoot * 50.0
                reward -= boundary_penalty

            # 8: Time Cost (Efficiency)
            reward -= 0.01

        self.current_reward = reward
        return reward

    def _calculate_overshoot(self, position: np.ndarray) -> float:
        """
        Calculate total distance outside workspace bounds.

        Returns sum of overshoot in each dimension (meters).
        """
        x, y, z = position
        overshoot = 0.0

        # X dimension
        if x < self.workspace_limits["x"][0]:
            overshoot += self.workspace_limits["x"][0] - x
        elif x > self.workspace_limits["x"][1]:
            overshoot += x - self.workspace_limits["x"][1]

        # Y dimension
        if y < self.workspace_limits["y"][0]:
            overshoot += self.workspace_limits["y"][0] - y
        elif y > self.workspace_limits["y"][1]:
            overshoot += y - self.workspace_limits["y"][1]

        # Z dimension
        if z < self.workspace_limits["z"][0]:
            overshoot += self.workspace_limits["z"][0] - z
        elif z > self.workspace_limits["z"][1]:
            overshoot += z - self.workspace_limits["z"][1]

        return overshoot

    def _is_terminated(self, states: dict) -> bool:
        robot_key = sorted(states.keys())[0]
        pipette_pos = np.array(states[robot_key]["pipette_position"])

        out_of_bounds = not self._is_in_workspace(pipette_pos)
        stayed_at_target = (
            self._steps_at_target[0] >= self.dwell_steps
        )  # Use index 0 for first agent

        return stayed_at_target or out_of_bounds

    def _is_in_workspace(self, position: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if position is within workspace limits.

        Args:
            position: The (x, y, z) position to check
            tolerance: Buffer zone in meters (default 1cm) to allow slight overruns
        """
        x, y, z = position
        return (
            self.workspace_limits["x"][0] - tolerance
            <= x
            <= self.workspace_limits["x"][1] + tolerance
            and self.workspace_limits["y"][0] - tolerance
            <= y
            <= self.workspace_limits["y"][1] + tolerance
            and self.workspace_limits["z"][0] - tolerance
            <= z
            <= self.workspace_limits["z"][1] + tolerance
        )

    def _scale_value(self, val, limits):
        """Helper to scale a single value to [-1, 1]"""
        low, high = limits
        return 2 * (val - low) / (high - low) - 1

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Helper to scale a 3D point to [-1, 1] range."""
        return np.array(
            [
                self._scale_value(pos[0], self.workspace_limits["x"]),
                self._scale_value(pos[1], self.workspace_limits["y"]),
                self._scale_value(pos[2], self.workspace_limits["z"]),
            ]
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
    entry_point="npec_robotics.ot2_env:OT2ENV",
    kwargs={
        "num_agents": 1,
        "render_mode": "human",
        "target": [0.0, 0.0, 0.1],
        "max_steps": 1000,
    },
)
