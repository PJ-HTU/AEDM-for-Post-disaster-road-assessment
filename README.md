# AEDM: Attention-based Encoder-Decoder Model for Post-disaster Road Assessment Drone Routing
This repository implements the **AEDM (Attention-based Encoder-Decoder Model)** proposed in the paper *"Deep Reinforcement Learning for Drone Route Optimization in Post-Disaster Road Assessment"*, aiming to solve rapid drone routing optimization for post-disaster road damage assessment.

## Core Objective
Address the time sensitivity and complexity of post-disaster road assessment by:
- Generating high-quality drone routes within 1–2 seconds (vs. 100–2000s for traditional methods)
- Maximizing the collection of road damage information without domain-specific algorithm design
- Supporting multi-drone coordination and adaptability to diverse disaster scenarios

## Key Features
1. **Network Transformation**: Converts link-based routing problems into node-based formulations to eliminate ambiguity and reduce computational complexity.
2. **Attention-based Encoder-Decoder Architecture**: Leverages Transformer to learn optimal routing strategies end-to-end via deep reinforcement learning.
3. **Multi-task Learning**: Handles diverse parameter combinations (drone count, assessment time limits) and generalizes to unseen scenarios.
4. **Rapid Inference**: Outperforms commercial solvers (20–71% improvement) and traditional heuristics (23–35% improvement) in solution quality.
5. **Synthetic Data Generation**: Solves large-scale training dataset scarcity by generating realistic road network instances.

## Model Architecture
![AEDM Architecture](https://raw.githubusercontent.com/PJ-HTU/AEDM-for-Post-disaster-road-assessment/main/Model%20Architecture.jpg)
The model processes road network coordinates, drone parameters, and constraints through:
- **Encoder**: Embeds node features and global parameters (drone count, time limits) via multi-head attention layers.
- **Decoder**: Sequentially constructs feasible routes with masking mechanisms to enforce time/battery constraints and avoid redundant assessments.

## Quick Start
### Dependencies
- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, Matplotlib
- Scipy

### Training
```bash
# Train on 100-node synthetic road network instances
python train_n100.py --epochs 200 --batch_size 64 --embedding_dim 128
```

### Testing
```bash
# Test on custom road network instances (supports up to 1000 nodes)
python test_n100.py --model_path ./checkpoints/aedm_best.pth --augmentation 8
```

## Citation
If you use this code or model in your research, please cite the paper:
```
@article{gong2024aedm,
  title={Deep Reinforcement Learning for Drone Route Optimization in Post-Disaster Road Assessment},
  author={Gong, Huatian and Sheu, Jiuh-Biing and Wang, Zheng and Yang, Xiaoguang and Yan, Ran},
  journal={[Journal Name]},
  year={2024},
  publisher={Elsevier/Springer/ACM}
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
