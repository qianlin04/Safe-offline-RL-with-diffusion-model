# Safe Offline Reinforcement Learning with Real-Time Budget Constraints

This is the Trajectory-based REal-time Budget Inference (TREBI) implementation from Safe Offline Reinforcement Learning with Real-Time Budget. This implementation is mainly based the Janner's [diffuser](https://github.com/jannerm/diffuser).


## Installation

```
conda env create -f environment.yml
conda activate TREBI
pip install -e .
```



## Tasks name

Pendulum and Reacher

- SafePendulum-medium-replay-v0
- SafeReacher-medium-replay-v0

MuJoCo tasks

- hopper-medium-v2
- hopper-medium-replay-v2
- hopper-medium-expert-v2
- walker2d-medium-v2
- walker2d-medium-replay-v2
- walker2d-medium-expert-v2
- halfcheetah-medium-v2
- halfcheetah-medium-replay-v2
- halfcheetah-medium-expert-v2



## Offline dataset generation

For MuJoCo tasks, we use the existing dataset from D4RL.

For Pendulum and Reacher:

```
python scripts/data_generate.py --dataset SafePendulum-v0 --datatype medium-replay --total_step 20000

python scripts/data_generate.py --dataset SafeReacher-v0 --datatype medium-replay --total_step 200000
```



## Training 

1. Train a diffusion model with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

Use the flag **--config config.toy_safe_env** for Pendulum and Reacher, and  **--config config.locomotion**  for MuJoCo tasks.

The default hyperparameters are listed in ./config/locomotion.py for MuJoCo tasks and ./config/toy_safe_env.py for Pendulum and Reacher.

2. Train a reward value function and a cost value function with:
```
python scripts/train_values.py --dataset halfcheetah-medium-expert-v2
```
and 

```
python scripts/train_cost_values.py --dataset halfcheetah-medium-expert-v2
```

Use the flag **--config config.toy_safe_env** for Pendulum and Reacher, and  **--config config.locomotion**  for MuJoCo tasks.


3. Evaluation:
```
python scripts/plan_cost_guided.py --dataset halfcheetah-medium-expert-v2 --ratio_of_maxthreshold 1.0
```
Use the **--ratio_of_maxthreshold $ratio** to control the budget ratio, for example **--ratio_of_maxthreshold 0.2** means budget ratio is set to 0.2.

Use flag **--config config.toy_safe_env** for Pendulum and Reacher, and  **--config config.locomotion**  for MuJoCo tasks.



## Using pretrained models

Diffuser provides several pretrained models of MuJoCo tasks, which can be directly used in our real-time budget constraint scenario. You can download these pretrained models and only train the cost value function, and then evaluate the algorithm.

### Downloading weights

Download pretrained diffusion models and value functions with:

```
./scripts/download_pretrained.sh
```

This command downloads and extracts a [tarfile](https://drive.google.com/file/d/1srTq0OFQtWIv9A7fwm3fwh1StA__qr6y/view?usp=sharing) containing [this directory](https://drive.google.com/drive/folders/1ie6z3toz9OjcarJuwjQwXXzDwh1XnS02?usp=sharing) to `logs/pretrained`. The models are organized according to the following structure:

```
└── logs/pretrained
    ├── ${environment_1}
    │   ├── diffusion
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       ├── sample-${epoch}-*.png
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   └── values
    │       └── ${experiment_name}
    │           ├── state_${epoch}.pt
    │           └── {dataset, diffusion, model, render, trainer}_config.pkl
    ├── ${environment_2}
    │   └── ...
```

The `state_${epoch}.pt` files contain the network weights and the `config.pkl` files contain the instantation arguments for the relevant classes.
The png files contain samples from different points during training of the diffusion model.

