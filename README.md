<h1 align="center">Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge (NeurIPS 2025)</h1>

<div align="center">

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2505.11197)

</div>


## Introduction
Modeling the dynamics from sparsely time-resolved snapshot data is crucial for understanding complex cellular processes and behavior. Existing methods leverage optimal transport, Schrödinger bridge theory, or their variants to simultaneously infer stochastic, unbalanced dynamics from snapshot data. However, these approaches remain limited in their ability to account for cell-cell interactions. This integration is essential in real-world scenarios since intercellular communications are fundamental life processes and can influence cell state-transition dynamics. To address this challenge, we formulate the Unbalanced Mean-Field Schrödinger Bridge (UMFSB) framework to model unbalanced stochastic interaction dynamics from snapshot data. Inspired by this framework, we further propose **CytoBridge**, a deep learning algorithm designed to approximate the UMFSB problem. By explicitly modeling cellular transitions, proliferation, and interactions through neural networks, CytoBridge offers the flexibility to learn these processes directly from data. The effectiveness of our method has been extensively validated using both synthetic gene regulatory data and real scRNA-seq datasets. Compared to existing methods, CytoBridge identifies growth, transition, and interaction patterns, eliminates false transitions, and reconstructs the developmental landscape with greater accuracy.

<p align="center">
  <img src="figs/overview.png" alt="overview" width="500">
</p>

## Environment Setup

It is recommended to use conda for environment management.

Create and activate a new environment:

```bash
conda env create -f environment.yml
conda activate CytoBridge
```

CytoBridge has been tested on Linux systems with CUDA available.

## Running Training Scripts

All training scripts are located in the `training/` folder. You can run different scripts depending on your dataset or task.

### Example

To reproduce our results on synthetic gene dataset with attractive interactions, simply run `train_simulation.py`:

```bash
cd training
python train_simulation.py
```

Other available training scripts:
- EMT: `train_emt.py`
- Mouse Hematopoiesis: `train_mouse.py`
- Pancreatic β-cell differentiation: `train_veres.py`
- Embryoid body: `train_eb.py`
- Zebrafish: `train_zebrafish.py`

You can run them in a similar way:

```bash
cd training
python <script_name>.py
```

We also provide trained checkpoints on these datasets in the `checkpoints/` folder

## Future Development
Please be aware that the functionality of this repository is scheduled to be merged into [CytoBridge](https://github.com/zhenyiizhang/CytoBridge), a comprehensive and user-friendly toolkit for dynamical optimal transport that we are actively developing. We recommend keeping an eye on the project for future updates. To use CytoBridge without cell-cell interactions, please check out [DeepRUOTv2](https://github.com/zhenyiizhang/DeepRUOTv2) 

## How to cite

If you find this package helpful in your research, we would greatly appreciate it if you could consider citing our following work. We would first like to recommend our new package CytoBridge (https://github.com/zhenyiizhang/CytoBridge), a comprehensive and user-friendly toolkit for dynamical optimal transport and spatiotemproal transcriptomic data that we are actively developing.

The first two papers are our surveys.

- Zhenyi Zhang, Zihan Wang, Yuhao Sun, Jiantao Shen, Qiangwei Peng, Tiejun Li, and Peijie Zhou. “Deciphering cell-fate trajectories using spatiotemporal single-cell transcriptomic data“.  *npj Syst Biol Appl 2025*. (https://www.nature.com/articles/s41540-025-00624-9) 
- Zhenyi Zhang, Yuhao Sun, Qiangwei Peng, Tiejun Li, and Peijie Zhou. “Integrating Dynamical Systems Modeling with Spatiotemporal scRNA-Seq Data Analysis”. In: *Entropy* 27.5, 2025b. ISSN: 1099-4300.

These papers present the core algorithm on which this package is built, as well as other relevant developments.

- Zhenyi Zhang, Tiejun Li, and Peijie Zhou. “Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport”. In: *ICLR 2025 Oral*.
- Zhenyi Zhang, Zihan Wang, Yuhao Sun, Tiejun Li, and Peijie Zhou. “Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge”. In: *NeurIPS 2025*.
- Dongyi Wang, Yuanwei Jiang, Zhenyi Zhang, Xiang Gu, Peijie Zhou, and Jian Sun. “Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling”. In: *NeurIPS 2025*.

Additional related papers may be cited as needed.
- Yuhao Sun, Zhenyi Zhang, Zihan Wang, Tiejun Li, and Peijie Zhou. “Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action”. In: *NeurIPS 2025*.
- Qiangwei Peng, Peijie Zhou, and Tiejun Li. “stVCR: Reconstructing spatio-temporal dynamics of cell development using optimal transport”. In: *Nature Methods*.

