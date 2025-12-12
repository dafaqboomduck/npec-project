from .sim_class import Simulation
import numpy as np

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)  # For one robot

# Initialize list to store the movement envelope (min/max bounds) for each axis
envelope = []

# Define velocities to test in both directions
velocities = [-0.2, 0.2]

# Define coordinate axes to explore
coords = ["x", "y", "z"]

# Iterate through each coordinate axis
for n, coord in enumerate(coords):
    # Initialize min and max coordinate values
    min_coord = np.inf
    max_coord = -np.inf
    
    # Test movement in both positive and negative velocity directions
    for velocity in velocities:
        # Create action array with velocity applied only to current axis
        actions = [velocity if i == n else 0 for i in range(len(coords))]
        actions.append(0)  # Add a fourth element for drop action
        actions = [actions]  # Wrap in list for sim.run() format

        if velocity < 0:
            # For positive velocity, continuously move and track minimum coordinate
            while True:
                state = sim.run(actions, num_steps=1)
                pipette_pos = state["robotId_1"]["pipette_position"]
                final_axis_coord = pipette_pos[n]
                if final_axis_coord < min_coord:
                    min_coord = final_axis_coord
                else:
                    break
        elif velocity > 0:
            # For negative velocity, continuously move and track maximum coordinate
            while True:
                state = sim.run(actions, num_steps=1)
                pipette_pos = state["robotId_1"]["pipette_position"]
                final_axis_coord = pipette_pos[n]
                if final_axis_coord > max_coord:
                    max_coord = final_axis_coord
                else:
                    break

    # Store the results: axis name and its movement bounds
    envelope.append([f"Axis_{coord}", min_coord, max_coord])

print("\n--- Summary ---")
print("Workspace limits for ot2_env.py:")
print(f'"x": ({envelope[0][1]:.4f}, {envelope[0][2]:.4f}),')
print(f'"y": ({envelope[1][1]:.4f}, {envelope[1][2]:.4f}),')
print(f'"z": ({envelope[2][1]:.4f}, {envelope[2][2]:.4f}),')