# AEDM: Attention-based Encoder-Decoder Model for Post-disaster Road Assessment Drone Routing
This repository implements the **AEDM (Attention-based Encoder-Decoder Model)** proposed in the paper *"Deep Reinforcement Learning for Drone Route Optimization in Post-Disaster Road Assessment"*, aiming to solve rapid drone routing optimization for post-disaster road damage assessment.

## Problem Statement

Given a road network affected by disaster, deploy a fleet of drones to:
- Maximize collected damage information across the road network
- Complete assessment within time constraints
- Respect battery flight time limits
- Determine optimal routes for multiple drones

## Model Architecture
![AEDM Architecture](https://raw.githubusercontent.com/PJ-HTU/AEDM-for-Post-disaster-road-assessment/main/Model%20Architecture.jpg)
The model processes road network coordinates, drone parameters, and constraints through:
- **Encoder**: Embeds node features and global parameters (drone count, time limits) via multi-head attention layers.
- **Decoder**: Sequentially constructs feasible routes with masking mechanisms to enforce time/battery constraints and avoid redundant assessments.

## Key Features

- **Rapid Performance**: 1-2 seconds inference time vs. 100-2,000 seconds for traditional methods
- **Superior Solution Quality**: over commercial solvers (Gurobi) and traditional heuristics
- **No Domain Expertise Required**: Eliminates need for hand-crafted algorithms through end-to-end learning
- **Strong Generalization**: Robust performance across varying problem scales, drone numbers, and time constraints
- **Multi-task Learning**: Handles diverse parameter combinations in a unified framework

## Technical Highlights

### Key Innovations
1. **Network Transformation**: Converts link-based routing problems (assessing road segments) into node-based formulations to eliminate ambiguity and reduce computational complexity .
2. **Synthetic Data Generation**: Addresses large-scale training dataset scarcity by generating realistic road network instances (grid initialization → link pruning → node perturbation) .
3. **Attention-based Encoder-Decoder**: Uses Transformer architecture to learn optimal routing strategies end-to-end via deep reinforcement learning (DRL) .
4. **Multi-task Learning**: Handles simultaneous training across varying drone numbers and time constraints, eliminating the need for separate models per parameter combination .


## Quick Start

### Dependencies
```bash
# Python environment requirements
Python 3.8+
PyTorch 1.10+
NumPy
Pandas
Matplotlib
Scipy
```

### Installation
```bash
git clone https://github.com/PJ-HTU/UM_PDRA.git
cd UM_PDRA
pip install -r requirements.txt
```

### Training
```bash
# Train on 100-node synthetic road network instances
python train_n100.py --epochs 200 --batch_size 64 --embedding_dim 128
```

**Training Parameters:**
- `epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 64)
- `embedding_dim`: Embedding dimension (default: 128)

### Testing
```bash
# Test on custom road network instances (supports up to 1000 nodes)
python test_n100.py --model_path ./checkpoints/um_best.pth --augmentation 8
```

**Testing Parameters:**
- `model_path`: Path to pre-trained model
- `augmentation`: Instance augmentation factor (improves solution diversity through coordinate flipping/swapping)

## Citation
If you use this code or model in your research, please cite the paper:
```
@article{gong2025deep,
  title={Deep Reinforcement Learning for Real-Time Drone Routing in Post-Disaster Road Assessment Without Domain Knowledge},
  author={Gong, Huatian and Sheu, Jiuh-Biing and Wang, Zheng and Yang, Xiaoguang and Yan, Ran},
  journal={arXiv preprint arXiv:2509.01886},
  year={2025}
}
```

## Repository Structure
The repository is organized to separate core logic (environment, model, training) from auxiliary tools, ensuring clarity and maintainability. 
```
AEDM-for-Post-disaster-road-assessment/
├── AEDM/                     # Core code directory (implements all model & task logic)
│   ├── PDRA/                 # Post-disaster Road Assessment (PDRA) task module
│   │   ├── POMO/             # Policy Optimization with Multiple Optima (POMO) implementation
│   │   ├── PDRAEnv.py        # PDRA environment class: simulates post-disaster road network scenarios
│   │   │   - Initializes dual networks (original road network for assessment + fully connected auxiliary network for transit).
│   │   │   - Implements environment interaction: `reset()` (reset scenario), `step()` (execute drone action and update state), and time/battery constraint checks.
│   │   │   - Calculates road link assessment time, transit time, and information value collection.
│   │   ├── PDRAModel.py      # AEDM model class: defines attention-based encoder-decoder architecture
│   │   │   - Encoder: Processes node features (coordinates, information value) and global parameters (K, p_max, Q) into high-dimensional embeddings via Transformer layers.
│   │   │   - Decoder: Sequentially generates drone routes using MHA, single-head attention (SHA), and masking (blocks infeasible actions like re-visiting information nodes).
│   │   │   - Outputs route probability distributions and ensures feasible solutions (e.g., drones return to depot within time limits).
│   │   ├── PDRATrainer.py    # Model training logic class
│   │   │   - Loads training instances (synthetic road networks) and initializes model/optimizer.
│   │   │   - Implements POMO-based training: multi-optima sampling, EMA-Z-score reward normalization (stabilizes multi-task training).
│   │   │   - Tracks training metrics (loss, collected information value) and saves checkpoints.
│   │   └── PDRATester.py     # Model testing logic class
│   │       - Loads pre-trained models and test instances (synthetic/real-world road networks like Anaheim).
│   │       - Evaluates model performance: calculates solution quality (collected information value), inference time.
│   │       - Supports 8-fold instance augmentation (coordinate flipping/swapping) to improve solution diversity.
│   └── utils/                # Auxiliary tools directory (supports core logic execution)
│       ├── utils.py          # General utility functions
│       │   - Log data management: `LogData` class to record training/testing metrics (loss, score, time) for visualization.
│       │   - Distance calculation: Computes Euclidean distance between nodes (for transit/assessment time estimation).
│       └── log_image_style/  # Log image styling configuration
│           └── style_PDRA_20.json # Defines visualization styles.
├── train_n100.py             # Training entry script (for 100-node synthetic instances)
│   - Defines hyperparameters: embedding dimension (128), encoder layers (6), batch size (64), epochs (200).
│   - Calls `PDRATrainer` to start training: samples synthetic instances, runs POMO training, and saves checkpoints to `checkpoints/`.
├── test_n100.py              # Testing entry script (for 100-node instances, extendable to 1000-node)
│   - Loads pre-trained models from `checkpoints/` and test instances (synthetic or real-world like Anaheim).
│   - Calls `PDRATester` to evaluate performance: outputs inference time, collected information value.
└── checkpoints/              # Pre-trained model storage directory
```
## Acknowledgements
💡 Our code builds on [POMO](https://github.com/yd-kwon/POMO). Big thanks! 
