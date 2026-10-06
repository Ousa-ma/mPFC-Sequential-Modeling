# mPFC Sequential Modeling


### Closed-loop recurrent network models of generalized task-progress sequences in the medial prefrontal cortex.  
This repository contains the network-modelling code, the precomputed results, and the figure notebooks.  
The analysis code for the experimental (calcium-imaging) data is in a separate [repository](JFSauer/task_progress_sequences).  
The imaging data can be found [here](https://doi.org/10.5281/zenodo.10528244). 


### Overview
In mice performing an odour-guided two-choice task, mPFC populations form activity sequences that run through cue sampling, movement and reward. These sequences share a common temporal structure across trial types, which we call generalized task-progress tuning. This repository asks whether that shared structure has a computational benefit.  


### Task 
A recurrent network has to reproduce K two-dimensional target trajectories. Each trajectory has three phases that mirror the behavioural task: a stationary sampling phase at the origin, a condition-specific curved movement phase, and a stationary reward phase at the endpoint. A single scalar cue c<sub>k</sub> tells the network which trajectory to produce.  

* **Control network (α = 0):** The network is trained only to reproduce the trajectories. Its hidden units spontaneously form cue-locked sequences that are shared across conditions.
  
* **Constrained network (α > 0):** Training also penalises the cross-condition correlation of each unit's tuning curve (P<sub>corr</sub>). This forces condition-specific representations.
  
* **Comparison:** Control and constrained networks are paired seed by seed: they share the connectivity mask, initial weights and batch order, and differ only in the penalty. Both reach the same final performance, but constrained networks need many more training steps. The gap grows with the number of conditions K.  


### Architecture and Training
The model is a discrete-time recurrent rate network trained in closed loop to perform the main task. All hyperparameters are defined in [`src/config.py`](src/config.py); a complete description is provided in the **Methods** section of the paper.


### Model configuration

| Component                        | Specification                                                                                                                                              |
| :---------------------------     | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture                 | Discrete-time recurrent rate network with **\(N = 200\)** ReLU units                                                                                       |
| Connectivity                 | Fixed sparse recurrent mask with **\(p = 0.4\)**, giving 16,000 active recurrent connections, sampled once per seed                                        |
| Input                        | Static cue c<sub>k</sub> concatenated with the network's previous ($\hat{x}$, $\hat{y}$) outputs (d<sub>in</sub> = 2T - 1 = 263)                         |
| Readout                      | Bias-free linear map producing ($\hat{x}$, $\hat{y}$)                                                                                                                |
| Training                     | Closed-loop, free-running (**no teacher forcing**); Adam, learning rate \(2\times10^{-4}\), batch size **2,000**, gradient clipping at global norm **1.0** |
| Regularization               | λ = (0.05, 0.005, 0.05)                                                                                                                      |
| Stopping criterion           | Threshold MSE  θ<sup>*</sup> = 0.06                                                                                                            |
| Trainable parameters         | 93,200                                                                                                                                                 |
| Seeds                        | 20 matched control/constrained pairs                                                                                                                   |

The training objective is:
```math
\mathcal{L} = L_{\mathrm{MSE}}
+ \lambda_1 \left( \lVert W_{\mathrm{in}} \rVert + \lVert W_{\mathrm{out}} \rVert \right)
+ \lambda_2 \, \lVert a \rVert 
+ \lambda_3 \lVert W_{\mathrm{rec}} \rVert
+ \alpha \, P_{\mathrm{corr}} .
```

Here, $`P_{\mathrm{corr}}`$ denotes the correlation penalty applied to the network activity across task conditions.                    


### Repository structure
```
mPFC-Sequential-Modeling/
├── src/                     # Model, data generation, analysis and plotting code
│   ├── config.py            #   All hyperparameters + kwargs dictionaries
│   ├── data.py              #   Synthetic diverging-trajectory task
│   ├── model.py             #   Model_1: closed-loop sparse RNN, correlation penalty, training loop
│   ├── loading.py           #   Weight-file naming convention, load_and_predict()
│   ├── analysis.py          #   RSM, epochs-to-criterion, RM-ANOVA helpers, dimensionality
│   ├── plotting.py          #   All figure functions
│   └── utils.py             #   Loss, activations, tf.data pipeline
├── notebooks/
│   ├── 1_train_and_evaluate.ipynb    # Train control vs constrained networks, evaluate them
│   └── 2_reproduce_paper_figs.ipynb  # Regenerate the paper's modelling figures
├── data/                    # Precomputed metrics (.pkl) used by notebook 2
├── results/
│   ├── models/              # Trained weights (.weights.h5), see results/models/README.md
│   └── figures/             # Exported figure panels
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── LICENSE
```

### Installation  
The code requires:  
* Python ≥ 3.9
* TensorFlow ≥ 2.16 (Keras 3)
* A **GPU** is recommended for full training runs.
* For the statistical analyses, `pingouin < 0.6` and `pandas < 3` are required.
All dependencies are pinned in [`requirements.txt`](requirements.txt).
Install them with:
```bash
pip install -r requirements.txt
```


### Notebooks
1. Train and evaluate (1_train_and_evaluate.ipynb)  
This notebook trains seed-matched pairs of control (α = 0) and constrained (α = 1) networks on the K = 5 trajectory task and saves their weights to results/models/. It then compares the two networks: predicted trajectories, sequential hidden-unit activity, representational similarity across conditions, and training speed.  

2. Reproduce the paper figures (2_reproduce_paper_figs.ipynb)  
This notebook regenerates important modelling figures:
* the emergence of shared task-progress sequences in control networks
* their loss in constrained networks
* the resulting learning cost
* dimensionality and weights
* the control analyses:
  - dataset size
  - trajectory angle
  - penalty strength



### Citation
If you use this code, please cite the paper:
```bibtex
@article{to add,
  title   = {Generalized prefrontal task-progress sequences efficiently encode task structure},
  author  = {Tarahi, Ousama and Koshy, Kevin and Sauer, Jonas-Frederic},
  journal = {to add},
  year    = {to add},
  doi     = {to add}
}
```

### Contact
* Ousama Tarahi : (ousama.tarahi@uni-saarland.de)  
* Jonas-Frederic Sauer -  Corresponding author : (jonas.sauer@uni-saarland.de).  
Centre for Integrative Physiology and Molecular Medicine, Saarland University, Homburg, Germany.
