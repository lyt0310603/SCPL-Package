---
title: 'DecoupleFlow: A python package for quickly refactor deep learning model to decouple architecture'
tags:
  - Python
  - machine learning
authors:
  - name: You-Teng Lin
    affiliation: 1
  - name: Hung-Hsuan Chen
    affiliation: 1
affiliations:
 - name: Data Analytics Research Team, National Central University, TW
   index: 1
date: 31 March 2026
bibliography: paper.bib
---

# Summary
<!-- A description of the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->
DecoupleFlow is a PyTorch-based package that helps users transform deep learning models into decoupled training architectures. It is designed to simplify the development and deployment of decoupled learning workflows, allowing users to configure model partitioning strategies, training components, and execution parameters through parameterized settings. By reducing the implementation complexity of decoupled learning methods, DecoupleFlow makes these techniques more accessible to researchers and practitioners.

# Statement of need
<!-- A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work. -->
Decoupled architectures are deep learning training methods that partition a model into multiple blocks and reduce gradient dependency across them, allowing more independent optimization. Such methods have been studied to address limitations of conventional end-to-end backpropagation, including high memory cost and limited parallelism.

Recent work, including Supervised Contrastive Parallel Learning (SCPL) and Decoupled Learning with Information Regularization (DeInfoReg), has shown the promise of this paradigm. However, these methods are often difficult to adopt because they require substantial model refactoring and customized configuration of block partitioning, projection heads, and loss functions.

DecoupleFlow was developed to reduce this implementation burden. It provides a parameterized framework for building decoupled training workflows, making these methods more accessible to researchers and practitioners and supporting reproducible experimentation.

# Software design  
<!-- An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description. -->
DecoupleFlow adopts a modular, parameterized design for decoupled deep learning training. Instead of requiring users to reimplement a full training pipeline for each method, the package organizes shared components into reusable modules. This architecture reduces implementation overhead and supports rapid construction and modification of decoupled training workflows.

DecoupleFlow transforms a user-defined model into a stack of DecoupleFlow Blocks. Each block contains three core components: an encoder, a projector, and a local loss module. This structure allows each block to optimize a local objective while preserving a training workflow that remains close to standard model development practice. In typical classification settings, the classifier is placed in the final block; accordingly, the final block does not attach a projector head and is updated with cross-entropy loss. In addition, DecoupleFlow provides an adaptive block variant; compared with the standard block design, each block includes an extra classifier to support early-exit condition evaluation during inference.

![DecoupleFlow Block](fig/DecoupleFlow_Block.jpg)
*Figure 1. DecoupleFlow Block containing an encoder, projector head, and local objective loss.*

These components are exposed through a parameterized interface, allowing users to configure decoupled workflows without extensive changes to their existing training code. Key configuration options include:
* `device_map`, which specifies how model blocks are assigned across devices;
* `loss_fn`, which specifies the local loss function;
* `projector_type`, which determines the projector head design;
* `transform_funcs`, which reshapes intermediate representations to avoid dimensional mismatches (e.g., in LSTM-based models); and
* additional parameters for optional features such as multithreaded execution and adaptive inference.

This design provides a common implementation framework for existing decoupled learning methods. In particular, DecoupleFlow modularizes key mechanisms from Supervised Contrastive Parallel Learning (SCPL) and Decoupled Learning with Information Regularization (DeInfoReg), enabling both workflows to be expressed within a unified software structure. As a result, the package improves reproducibility, supports method comparison, and provides a practical basis for extending decoupled learning in future research.

# Experiment
# AI usage disclosure
# Reference