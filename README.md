# ANF (adaptive normalizing flow)


ANF (Action-based Neural Flow) is a neural simulation framework designed to efficiently generate independent lattice field theory (LFT) configurations for complex target distributions. By learning invertible transformations from simple Gaussian inputs to target distributions defined by physical actions, ANF dramatically accelerates LFT sampling while reducing computational cost and autocorrelation times.

# Key Features

Efficient Sampling: Learns a transformation from a Gaussian distribution to complex target distributions defined by the Euclidean action.
Physics-Guided Training: Optimized using action-based objectives for accurate and physical sampling.
LoRA-based Mixer Model: Implements transformations using a low-rank adaptation (LoRA)-based mixer, with trainable patch embedding layers and alternating mixer blocks for patch and channel updates.

# Usage
Training: Train ANF using physics-guided, action-based optimization to learn the transformation from input distribution to target distribution.
Inference: Generate independent LFT configurations in parallel with reduced computational cost and improved sampling efficiency.
Fine-Tuning: Adapt to new lattice action parameters using LoRA-based fine-tuning for rapid transfer without retraining the entire model.
