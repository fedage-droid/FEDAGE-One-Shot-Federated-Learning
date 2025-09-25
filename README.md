# FEDAGE

## Setup

Make sure you have **Python 3.9.21** installed.

Install the required Python packages using:

```bash
pip3 install -r requirements.txt
```

## Running the Pipeline

You can run the pipeline with the default parameters by using the following shell script:

```bash
#!/bin/bash

# Run pipeline.py with default arguments

python pipeline.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --num_clients 10 \
    --alpha 0.1 \
    --output_dim 64 \
    --epochs 7 \
    --batch_size 64 \
    --lr 0.001 \
    --beta 0.5 \
    --temperature 2.0 \
    --seed 42 \
    --mixture_coef 0.75 \
    --session 0 \
    --dp_epsilon inf \
    --dp_delta 1e-5 \
    --dp_clip_norm 1.0
```

## Command-line Arguments

This pipeline supports several configurable arguments for dataset selection, model backbone, training setup, and optional differential privacy. Below are the available options:

### Dataset (`--dataset`)
Choose the dataset to train on:
- **cifar10**
- **cifar100**
- **stl10** 
- **mnist**
- **fashionmnist**

_Default: `cifar10`_

---

### Backbone (`--backbone`)
Select the neural network architecture:
- **resnet18** – Standard ResNet-18 CNN.
- **mobilenet** – Lightweight CNN optimized for mobile/edge.
- **squeezenet** – Compact CNN with fewer parameters.
- **vgg19** – Deeper VGG variant with 19 layers.

_Default: `resnet18`_

---

### Training and Federated Parameters
- **`--num_clients`** (int, default: `10`)  
  Number of participating clients in federated training.

- **`--alpha`** (float, default: `0.1`)  
  Dirichlet distribution parameter for data partitioning (controls heterogeneity).

- **`--output_dim`** (int, default: `64`)  
  Dimension of the model’s output representation.

- **`--epochs`** (int, default: `7`)  
  Number of training epochs per client per round.

- **`--batch_size`** (int, default: `64`)  
  Local batch size for client updates.

- **`--lr`** (float, default: `1e-3`)  
  Learning rate for local optimizers.

- **`--beta`** (float, default: `0.5`)  
  Auxiliary coefficient (e.g., for regularization or loss mixing).

- **`--temperature`** (float, default: `2.0`)  
  Softmax temperature for contrastive/distillation tasks.

- **`--mixture_coef`** (float, default: `0.75`)  
  Weighting coefficient for combining losses or updates.

- **`--session`** (int, default: `0`)  
  Session identifier for multi-run experiments.

- **`--seed`** (int, default: `None`)  
  Random seed for reproducibility.

---

### Differential Privacy
- **`--dp_epsilon`** (float, default: `inf`)  
  Privacy budget ε. Set to `inf` to disable DP.

- **`--dp_delta`** (float, default: `1e-5`)  
  Target δ for DP.

- **`--dp_clip_norm`** (float, default: `1.0`)  
  L2 clipping bound for latents before adding noise.
