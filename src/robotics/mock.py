import gymnasium as gym
import numpy as np
import ot2_env  # needed to trigger registration
from pid import PIDController


# Top corner
# top_corner = [0.10775, 0.088, 0.1215]
# Target positions
targets = [
    [0.1145, 0.0670, 0.1215],
    [0.1491, 0.1051, 0.1215],
    [0.1837, 0.1432, 0.1215],
    [0.2183, 0.1814, 0.1215],
    [0.2440, 0.1900, 0.1215],
]

# Separate PID controller for each axis
pid_x = PIDController(kp=9.0, ki=0.048, kd=0.0)
pid_y = PIDController(kp=9.0, ki=0.048, kd=0.0)
pid_z = PIDController(kp=5.0, ki=0.024, kd=0.0)

dt = 1.0 / 240.0
max_steps = 5000
drop_threshold = 0.0001  # Distance threshold to trigger drop
drop_cooldown = 50  # Steps to wait after dropping before moving to next target

env = gym.make("OT2ENV-v0", max_steps=max_steps)
obs, _ = env.reset(seed=42)

target_idx = 0
dropped_at_target = False
cooldown_counter = 0

print(f"Moving to target {target_idx}")

for step in range(max_steps):
    # Check if we've completed all targets
    if target_idx >= len(targets):
        print(f"All targets completed at step {step}!")
        break

    target = targets[target_idx]

    # Extract current pipette position (indices 6, 7, 8 in observation)
    x, y, z = obs[6:9]
    target_x, target_y, target_z = target

    # Calculate error for each axis
    error_x = target_x - x
    error_y = target_y - y
    error_z = target_z - z

    # Calculate distance
    distance = np.sqrt(error_x**2 + error_y**2 + error_z**2)

    # Compute PID control for each axis
    control_x = pid_x.compute(error_x, dt, output_limits=(-1.0, 1.0))
    control_y = pid_y.compute(error_y, dt, output_limits=(-1.0, 1.0))
    control_z = pid_z.compute(error_z, dt, output_limits=(-1.0, 1.0))

    # State machine logic
    if not dropped_at_target:
        # Still approaching or at target, haven't dropped yet
        if distance <= drop_threshold:
            # Close enough - drop the droplet
            action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            dropped_at_target = True
            cooldown_counter = 0
            print(
                f"Dropping at target {target_idx} (step {step}), distance: {distance:.6f}"
            )
            print(target_x, target_y)
        else:
            # Still approaching - keep moving
            action = np.array([control_x, control_y, control_z, 0.0], dtype=np.float32)
    else:
        # Already dropped, in cooldown phase
        cooldown_counter += 1
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if cooldown_counter >= drop_cooldown:
            # Cooldown complete - move to next target
            target_idx += 1
            dropped_at_target = False
            cooldown_counter = 0

            # Reset PID controllers for the new target
            pid_x.reset()
            pid_y.reset()
            pid_z.reset()

            print(f"Moving to target {target_idx}")

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

env.close()
print(f"Finished! Dropped at {target_idx} targets")
