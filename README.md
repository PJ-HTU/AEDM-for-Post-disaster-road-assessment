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
![AEDM Architecture](6.jpg)
The model processes road network coordinates, drone parameters, and constraints through:
- **Encoder**: Embeds node features and global parameters (drone count, time limits) via multi-head attention layers.
- **Decoder**: Sequentially constructs feasible routes with masking mechanisms to enforce time/battery constraints and avoid redundant assessments.

## Quick Start
### Dependencies
- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, Matplotlib
- Scipy (for distance calculation)

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

## Performance
| Metric                | AEDM Performance       | Traditional Methods |
|-----------------------|------------------------|---------------------|
| Inference Time        | 1–2 seconds            | 100–2000 seconds    |
| Solution Quality (vs. Commercial Solvers) | +20–71% | -                   |
| Solution Quality (vs. Heuristics) | +23–35% | -                   |
| Max Supported Nodes   | 1000                   | ≤ 600               |
| Domain Knowledge Requirement | No | Yes |

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
│   │   │   - Initializes dual networks (original road network for assessment + fully connected auxiliary network for transit) {insert\_element\_8\_}.
│   │   │   - Implements environment interaction: `reset()` (reset scenario), `step()` (execute drone action and update state), and time/battery constraint checks {insert\_element\_9\_}.
│   │   │   - Calculates road link assessment time, transit time, and information value collection {insert\_element\_10\_}.
│   │   ├── PDRAModel.py      # AEDM model class: defines attention-based encoder-decoder architecture
│   │   │   - Encoder: Processes node features (coordinates, information value) and global parameters (K, p_max, Q) into high-dimensional embeddings via Transformer layers {insert\_element\_11\_}.
│   │   │   - Decoder: Sequentially generates drone routes using MHA, single-head attention (SHA), and masking (blocks infeasible actions like re-visiting information nodes) {insert\_element\_12\_}.
│   │   │   - Outputs route probability distributions and ensures feasible solutions (e.g., drones return to depot within time limits) {insert\_element\_13\_}.
│   │   ├── PDRATrainer.py    # Model training logic class
│   │   │   - Loads training instances (synthetic road networks) and initializes model/optimizer {insert\_element\_14\_}.
│   │   │   - Implements POMO-based training: multi-optima sampling, EMA-Z-score reward normalization (stabilizes multi-task training) {insert\_element\_15\_}.
│   │   │   - Tracks training metrics (loss, collected information value) and saves checkpoints {insert\_element\_16\_}.
│   │   └── PDRATester.py     # Model testing logic class
│   │       - Loads pre-trained models and test instances (synthetic/real-world road networks like Anaheim) {insert\_element\_17\_}.
│   │       - Evaluates model performance: calculates solution quality (collected information value), inference time, and relative gap vs. baselines (Gurobi, GC+LS) {insert\_element\_18\_}.
│   │       - Supports 8-fold instance augmentation (coordinate flipping/swapping) to improve solution diversity {insert\_element\_19\_}.
│   └── utils/                # Auxiliary tools directory (supports core logic execution)
│       ├── utils.py          # General utility functions
│       │   - Log data management: `LogData` class to record training/testing metrics (loss, score, time) for visualization {insert\_element\_20\_}.
│       │   - Coordinate processing: Converts latitude/longitude to planar coordinates and normalizes to [0,1]² {insert\_element\_21\_}.
│       │   - Distance calculation: Computes Euclidean distance between nodes (for transit/assessment time estimation) {insert\_element\_22\_}.
│       └── log_image_style/  # Log image styling configuration
│           └── style_PDRA_20.json # Defines visualization styles (e.g., radar chart axes range, box plot color) for training/testing logs (used in Figure 8, 9) {insert\_element\_23\_}.
├── train_n100.py             # Training entry script (for 100-node synthetic instances)
│   - Defines hyperparameters: embedding dimension (128), encoder layers (6), batch size (64), epochs (200) {insert\_element\_24\_}.
│   - Calls `PDRATrainer` to start training: samples synthetic instances, runs POMO training, and saves checkpoints to `checkpoints/` {insert\_element\_25\_}.
├── test_n100.py              # Testing entry script (for 100-node instances, extendable to 1000-node)
│   - Loads pre-trained models from `checkpoints/` and test instances (synthetic or real-world like Anaheim) {insert\_element\_26\_}.
│   - Calls `PDRATester` to evaluate performance: outputs inference time, collected information value, and gap vs. baselines {insert\_element\_27\_}.
└── checkpoints/              # Pre-trained model storage directory
    └── aedm_best.pth         # Example pre-trained model file: saved after 200 epochs (achieves 1–2s inference and 4.07–5.13% gap vs. optimal) {insert\_element\_28\_}.
```
