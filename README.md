**This tool is still under continuous development. The operation instructions and related descriptions are for reference only.**

# SCPL-Package

SCPL-Package is a tool designed to accelerate the training of deep learning models. It allows users to input custom model architectures and efficiently distribute the internal layers across multiple GPUs for distributed training. Its core technologies include gradient truncation, local objectives, and parallel processing, which significantly enhance model training speed.

## Key Features

- **Dynamic Model Allocation**: Redistributes internal layers across multiple GPUs based on the user-provided model architecture.
- **Gradient Truncation**: Reduces inter-device communication overhead through gradient truncation techniques, improving training efficiency.
- **Local Objectives**: Optimizes local training performance using contrastive learning as the local objective.
- **Parallel Processing**: Supports multi-GPU parallel processing to further accelerate large-scale model training.

## Core Technologies

### **Introduction to SCPL**
Supervised Contrastive Parallel Learning (SCPL) decouples backpropagation (BP) through multiple local training objectives and supervised contrastive learning. It transforms the long gradient flow in traditional deep networks into multiple short gradient flows and implements pipeline-based parallel processing, enabling independent training of parameters at each layer. This method addresses the inefficiency caused by "backward locking" in backpropagation, achieving faster training speeds compared to conventional BP.  
GitHub: [Here](https://github.com/minyaho/SCPL/tree/main?tab=readme-ov-file)

### **Introduction to DeInfoReg**
Decoupled Supervised Learning with Information Regularization (DeInfoReg) enhances model performance and flexibility compared to SCPL by designing new local loss functions and model structures. The new model structure features Adaptive Inference Paths, Dynamic Expanded Layers, and Dynamic Extended Layers. This tool replicates only the new local loss function and Adaptive Inference Path.  
GitHub: [Here](https://github.com/ianzih/Decoupled-Supervised-Learning-for-Information-Regularization/tree/master)

## Usage
Detailed usage instructions (Chinese only): [Here](https://hackmd.io/@b3NdIM1JStCqtPPOfgoapw/HyM1ei860)
