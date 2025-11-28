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
```
AEDM-for-Post-disaster-road-assessment/
├── AEDM/
│   ├── PDRA/            # Core module (environment, model, trainer)
│   │   ├── POMO/        # Policy Optimization with Multiple Optima implementation
│   │   ├── PDRAEnv.py   # Environment simulation for post-disaster road assessment
│   │   ├── PDRAModel.py # Attention-based encoder-decoder model
│   │   └── PDRATrainer.py # Model training logic
│   └── utils/           # Tool functions (logging, image styling, data processing)
├── train_n100.py        # Training entry (100-node instances)
├── test_n100.py         # Testing entry (100-node instances)
└── checkpoints/         # Pre-trained model storage
```

要不要我帮你生成一份**环境配置脚本**和**示例数据文件**，方便快速部署和测试？
