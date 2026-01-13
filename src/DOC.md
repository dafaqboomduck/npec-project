# NPEC Automated Plant Root Inoculation Pipeline

## Overview

This project was developed for the **Netherlands Plant Eco-phenotyping Centre (NPEC)** to automate the process of inoculating plant root tips in Petri dishes. The system combines deep learning-based computer vision with robotic control to eliminate human error and significantly reduce processing time.

The pipeline processes images of *Arabidopsis thaliana* seedlings grown in Petri dishes, detects individual plant roots, measures their lengths, localizes root tips, and controls an OT-2 liquid handling robot to precisely deliver inoculant to each root tip.

### Key Capabilities

- **Root Segmentation**: U-Net based semantic segmentation achieving 88% F1 score
- **Anchor Detection**: Identifies seed/shoot locations to associate roots with individual plants
- **Adaptive Thresholding**: Dynamically adjusts detection sensitivity to find all plants
- **Geodesic Length Measurement**: Accurate root length calculation via skeletonization
- **Root Tip Localization**: Precise endpoint detection for robot targeting
- **Robotic Control**: PID and RL-based controllers for OT-2 manipulation
- **End-to-End Integration**: Seamless pipeline from image capture to inoculation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT IMAGE                                     │
│                    (Petri dish with 5 Arabidopsis plants)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CV PIPELINE (PlantMaster)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Petri     │  │  Inference  │  │ Segmentation│  │  Length Calculation │ │
│  │  Extraction │─▶│  (U-Net x2) │─▶│  & Anchors  │─▶│  (Skeletonization)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │     Root Tips (pixel coords)    │
                    │     Root Lengths (pixels)       │
                    │     Plant Masks & Skeletons     │
                    └─────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROBOTICS PIPELINE                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Coordinate │  │  Controller │  │    OT-2     │  │     Inoculation     │ │
│  │  Transform  │─▶│  (PID / RL) │─▶│ Environment │─▶│      Execution      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
src/
├── core/                      # Core processing algorithms
│   ├── plant_master.py        # Central pipeline orchestrator
│   ├── plant_segmenation.py   # Anchor detection & root assignment
│   └── root_lengths.py        # Geodesic length calculation
├── inference/                 # Model inference management
│   ├── inference_manager.py   # Patch-based inference pipeline
│   └── custom_functions.py    # Custom loss functions & metrics
├── robotics/                  # Robot control system
│   ├── controllers/           # Controller implementations
│   │   ├── base.py            # Abstract controller interface
│   │   ├── pid_controller.py  # PID-based controller
│   │   └── rl_controller.py   # Reinforcement learning controller
│   ├── ot2_env.py             # Gymnasium environment for OT-2
│   ├── pipeline.py            # Integrated robotics pipeline
│   └── sim_class.py           # PyBullet simulation wrapper
├── configs/                   # Configuration files
├── models/                    # Pydantic data models
├── helpers/                   # Utility functions
├── exceptions/                # Custom exceptions
├── artifacts/models/          # Trained model checkpoints
└── cv_pipeline.py             # High-level CV API wrapper
```

---

## Computer Vision Pipeline

### Inference System

The inference system uses patch-based processing to handle high-resolution Petri dish images efficiently.

#### Model Architecture

A U-Net encoder-decoder architecture optimized for binary segmentation:

| Component | Details |
|-----------|---------|
| Input | 512 × 512 × 3 (RGB patches) |
| Encoder | 4 blocks: 32 → 64 → 128 → 256 → 512 filters |
| Decoder | 4 blocks with skip connections |
| Normalization | Batch normalization after each conv layer |
| Activation | ReLU (hidden), Sigmoid (output) |
| Regularization | Dropout between blocks |
| Parameters | ~7.8M |
| Output | 512 × 512 × 1 (probability mask) |

#### Training Strategy

The model was **initially trained using knowledge distillation**, followed by **full end-to-end retraining without any layer freezing**:

- **Teacher Models**: Ensemble of root and shoot detection models  
- **Training Data (Distillation Phase)**: 50% ground truth labels + 50% teacher predictions  
- **Retraining Phase**: Full retraining on 100% ground truth images, allowing all parameters to update  
- **Initialization**: Retraining started from the distilled weights, providing a strong parameter initialization while allowing suboptimal representations to be overwritten  
- **Loss Function**: Weighted Dice–BCE (40% Dice, 60% BCE)  
- **Optimizer**: Adam (lr = 5e-4) with ReduceLROnPlateau  
- **Regularization**: Early stopping (patience = 7)  
- **Performance**: 88% F1 score on the validation set  

This approach leveraged teacher-guided pretraining to accelerate convergence while ensuring the final model learned exclusively from ground truth data.


#### Custom Loss Function

```python
def dice_bce_loss(y_true, y_pred):
    dice_weight, bce_weight = 0.4, 0.6
    dice_loss = tf.keras.losses.dice(y_true, y_pred)
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return (dice_weight * dice_loss) + (bce_weight * bce_loss)
```

The weighted combination addresses class imbalance (roots occupy small fraction of image).

#### Inference Pipeline

```
Input Image (4000+ × 3000+ px)
        │
        ▼
┌───────────────────┐
│ Extract Petri Dish│  ← Circular region detection
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Pad to Multiple │  ← Ensure divisibility by patch size
│   of Patch Size   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Patchify       │  ← Split into 512×512 non-overlapping patches
│   (N × 512 × 512) │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Model Inference  │  ← Batch prediction on all patches
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Unpatchify     │  ← Reconstruct full probability map
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Restore to       │  ← Map back to original image coordinates
│  Original Coords  │
└───────────────────┘
```

### Plant Segmentation

#### Anchor Detection

Anchors (seed+shoot locations) are used to associate root components with individual plants:

1. **Component Detection**: Connected component analysis on anchor mask
2. **Filtering**: Remove components below minimum area or outside top region
3. **Zone Assignment**: Assign to 1 of 5 plant zones based on x-coordinate
4. **Deduplication**: Keep only the largest anchor per zone

#### Adaptive Threshold Detection

The system dynamically adjusts the confidence threshold to ensure all plants are detected:

```python
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

for threshold in thresholds:
    results = find_endpoints(root_probs, anchors, threshold)
    if all_plants_detected(results):
        return threshold, results
```

#### Shoot Overlap Penalty

Prevents false positive root detections where shoots overlap:

```python
adjusted_probs[overlap_region] = root_probs[overlap_region] - penalty
```

### Root Length Calculation

Geodesic (along-the-root) length measurement using graph-based pathfinding:

#### Algorithm

1. **Skeletonization**: Reduce binary mask to 1-pixel-wide skeleton
2. **Gap Bridging**: Connect nearby endpoints using line drawing
3. **Graph Construction**: Build weighted graph with 8-connectivity
   - Cardinal neighbors: weight = 1.0
   - Diagonal neighbors: weight = √2 ≈ 1.414
4. **Pathfinding**: Dijkstra's algorithm from start to end point
5. **Length**: Sum of edge weights along shortest path

#### Gap Bridging

Handles discontinuities in the skeleton:

```python
def bridge_skeleton_gaps(skeleton, max_gap, max_path):
    endpoints = find_endpoints(skeleton)
    tree = cKDTree(endpoints)
    
    for endpoint in endpoints:
        nearby = tree.query(endpoint, distance_upper_bound=max_gap)
        if not already_connected(endpoint, nearby, max_path):
            draw_line(skeleton, endpoint, nearby)
    
    return skeleton
```

---

## Robotics Pipeline

### OT-2 Gymnasium Environment

A custom Gymnasium environment wrapping a PyBullet simulation of the OT-2 liquid handling robot.

#### State Space

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Joint Velocities | 3 | Current velocity of X, Y, Z joints |
| Pipette Position | 3 | Normalized (x, y, z) position |
| Relative Vector | 3 | Vector from pipette to target |
| Target Position | 3 | Normalized target coordinates |
| **Total** | **12** | Per agent (9) + shared target (3) |

#### Action Space

Continuous control over pipette velocity:
- **X velocity**: [-1.0, 1.0]
- **Y velocity**: [-1.0, 1.0]
- **Z velocity**: [-1.0, 1.0]

#### Reward Function

A multi-component reward designed for stable, precise manipulation:

| Component | Description | Weight |
|-----------|-------------|--------|
| Distance Improvement | Reward for moving closer to target | Dense |
| Proximity Bonus | Exponential reward near target | exp(-d/0.01) × 0.2 |
| Velocity Penalty | Penalize fast movement near target | -0.3 × v × proximity |
| Dwell Reward | Reward for staying at target | 0.3 + 0.7 × progress |
| Milestone Bonuses | One-time rewards at 10mm, 5mm, 2mm, 1mm | +1.0 to +4.0 |
| Success Bonus | Completing dwell requirement | +20.0 |
| Boundary Penalty | Leaving workspace | -3.0 - 50 × overshoot |
| Time Cost | Efficiency incentive | -0.01/step |

### Controller Architecture

#### Base Controller

Abstract interface supporting multi-target sequencing:

```python
class BaseController(ABC):
    def step(self, obs, info, targets, dt):
        """Returns (action, done, completed_target_idx)"""
        
        if at_target and dwell_complete:
            advance_to_next_target()
        else:
            action = self.compute_action(obs, info, target, dt)
        
        return action, all_complete, just_completed
```

#### PID Controller

Three independent PID controllers for X, Y, Z axes:

```python
class PIDRobotController(BaseController):
    def compute_action(self, obs, info, target, dt):
        error = target - current_position
        
        control_x = pid_x.compute(error[0], dt)
        control_y = pid_y.compute(error[1], dt)
        control_z = pid_z.compute(error[2], dt)
        
        return [control_x, control_y, control_z]
```

#### RL Controller

Uses a trained PPO model from Stable Baselines3:

```python
class RLController(BaseController):
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
    
    def compute_action(self, obs, info, target, dt):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
```

### Integrated Pipeline

The `RoboticsPipeline` class orchestrates the complete workflow:

```python
pipeline = RoboticsPipeline(
    inference_config=INFERENCE_CONFIG,
    segmentation_config=SEGMENTATION_CONFIG,
    env_config=ENV_CONFIG,
    controller_config=CONTROLLER_CONFIG,
)

# Run complete manipulation sequence
num_completed = pipeline.run_robotic_manipulation(
    starting_point=(0.10775, 0.062)
)
```

#### Execution Flow

1. **Image Capture**: Get current plate image from robot camera
2. **Root Detection**: Process through PlantMaster pipeline
3. **Coordinate Transform**: Convert pixel → mm → robot workspace (meters)
4. **Multi-Target Loop**:
   - Approach target using controller
   - Dwell at target for stability confirmation
   - Dispense droplet
   - Advance to next target
5. **Completion**: Log accuracy metrics

---

## Usage

### CV Pipeline Only

```python
from src.cv_pipeline import CVPipeline

pipeline = CVPipeline(config)
results = pipeline.process_directory("path/to/images/")

for result in results:
    print(f"Image: {result.image_name}")
    for plant_id, plant in result.plant_results.items():
        if plant.detected:
            print(f"  Plant {plant_id}: length={plant.length}px, tip={plant.end_point}")
```

### Full Robotics Pipeline

```python
from src.robotics.pipeline import RoboticsPipeline
from src.robotics.controllers import PIDRobotController, RLController

# Using PID controller
pipeline = RoboticsPipeline(...)
pipeline.run_robotic_manipulation(starting_point=(0.10775, 0.062))

# Using RL controller
rl_controller = RLController(model_path="artifacts/models/rl_model.zip")
pipeline.set_controller(rl_controller)
pipeline.run_robotic_manipulation(starting_point=(0.10775, 0.062))
```

### Command Line

```bash
# Run with PID controller (default)
python -m src.robotics.pipeline # --controller pid

# Run with RL controller
python -m src.robotics.pipeline --controller rl --model-path artifacts/models/rl_model.zip
```

---

## Dependencies

### Core

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy
- SciPy
- scikit-image
- NetworkX
- Patchify

### Robotics

- Gymnasium
- PyBullet
- Stable Baselines3 (for RL controller)

---

## Results

| Metric | Value |
|--------|-------|
| Root Segmentation F1 | 88% |
| Root Tip Localization | Sub-pixel accuracy |
| Robot Positioning Accuracy | < 1mm |
| End-to-End Processing Time | ~7s per plate |

---

## Future Improvements

- [ ] Real-time inference optimization (TensorRT/ONNX)
- [ ] Multi-robot coordination for parallel processing
- [ ] Active learning for model improvement
- [ ] Web-based monitoring dashboard
- [ ] Integration with NPEC's system

---

## Acknowledgments

Developed for the Netherlands Plant Eco-phenotyping Centre (NPEC) as part of the FAI2-ADSAI program. The project addresses the Hades module's need for automated, high-throughput plant-microbe interaction phenotyping.