# ANF (adaptive normalizing flow)


ANF is a neural simulation framework designed to efficiently generate independent lattice field theory (LFT) configurations for complex target distributions. By learning invertible transformations from simple Gaussian inputs to target distributions defined by physical actions, ANF dramatically accelerates LFT sampling while reducing computational cost and autocorrelation times.

# Key Features

**Efficient Sampling:** Learns a transformation from a Gaussian distribution to complex target distributions defined by the Euclidean action.

**Physics-Guided Training:** Optimized using action-based objectives for accurate and physical sampling.

**LoRA-based Mixer Model:** Implements transformations using a low-rank adaptation (LoRA)-based mixer, with trainable patch embedding layers and alternating mixer blocks for patch and channel updates.

# Important Modules:

**Layers/mixer.py** Defines the mixer model.

**Layers/affine_couplings.py** Implements reversible coupling layers using the mixer model.

**Layers/flow.py** Builds normalizing flows using reversible coupling layers to map Gaussian distributions to the target distribution.

**Layers/phi4.py** Defines the action of the φ4 field and some physical properties.

**Layers/MCMC.py** Performs the MCMC procedure.

**Layers/lora.py** Defines the LoRA layers.

**Training/Loss.py** Defines the loss function.

**Training/train.py** Defines the training procedure.


# Usage
**Training:** Train ANF using physics-guided, action-based optimization to learn the transformation from input distribution to target distribution.
“python train.py”   During training, the **loss** decreases while the **ESS** increases. By default, the training state_dict is saved as “phi4_rt_12x12_base.zip”.

**Fine-Tuning:** Adapt to new lattice action parameters using LoRA-based fine-tuning for rapid transfer without retraining the entire model. Meanwhile, a programming error was introduced into the analogue weights to simulate the compensatory effect of LoRA (digital) on analogue programming errors.

“python lora-train.py”  By default, "phi4_rt_12x12_base.zip" is loaded. Noise is added to the analogue weights, including the fully connected layers and convolutional layers. These weights are then fixed, and only the LoRA weights are trained. Additionally, the trained **state_dict** is automatically saved as "phi4_rt_12x12_lora.zip".


**Inference:** Generate independent LFT configurations in parallel with reduced computational cost and improved sampling efficiency.

“python inference.py”  By default, "phi4_rt_12x12_lora.zip" is loaded to generate configurations. Then, the MCMC procedure is executed, and the sampled configurations are used to compute physical properties such as susceptibility, two-point correlation functions, Ising energy, and others.


**More content will be gradually added before the publication of the article.**
