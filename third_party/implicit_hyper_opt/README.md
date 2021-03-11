Commentary Parameter Learning with IFT
======

This folder is adapted from the repository: [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://github.com/lorraine2/implicit-hyper-opt/). The original paper describing these algorithms was: [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/abs/1911.02590).


The full original README is included below for reference.  Please refer to the higher level README in this repository for details on running experiments.


-------------------------
-------------------------



# Optimizing Millions of Hyperparameters by Implicit Differentiation

This repository is an implementation of [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/abs/1911.02590).

# Running Experiments


## Setup Environment

Create a Python 3.7 environment and install required packages:
```bash
conda create -n ift-env python=3.7
source activate ift-env
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install -r requirements.txt
```

## Simple test

Consider the following tests to verify the environment is correctly setup:

### mnist_test.py
```
python mnist_test.py 
  --datasize <train set size> 
  --valsize <validation set size> 
  --lrh <hyperparameter lr need to be negative> 
  --epochs <min epochs for training model> 
  --hepochs <# of iterations for hyperparameter update> 
  --l2 <initial log weight decay> 
  --restart <reinitialize model weight after each hyperparameter update or not> 
  --model <cnn for lenet like model, mlp for logistic regession and mlp>
  --dataset <CIFAR10 or MNIST>
  --num_layers <# of hidden layer for mlp>
  --hessian<KFAC: KFAC estiamte; direct:true hessian and inverse>
  --jacobian<direct: true jacobian; product: use d_L/d_theta * d_L/d_lambda>
```

Trained models after each hyperparameter update will be stored in folder difined in line 627 in mnist_test.py
To use CG to compute inverse of hessian, change line 660's hyperparameter updator.

```bash
python mnist_test.py --datasize 40000 --valsize 10000 --lrh 0.01 --epochs=100 --hepochs=10 --l2=1e-5 --restart=10 --model=mlp --dataset=mnist --num_layers=1 --hessian=kfac --jacobian=direct
```

## Deployment

First, make sure you are on the master node:
```bash
ssh <USERNAME>@q.vectorinstitute.ai
```

Submit a job to the Slurm scheduler:
```bash
srun --partion=gpu --gres=gpu:1 --mem=4GB python mnist_test.py
```

Or, submit a batch of jobs defined by `srun_script.sh`:
```bash
sbatch --array=0-2 srun_script.sh
```

View queued jobs for a user:
```bash
squeue -u $USERNAME
```

Cancel jobs for a user:
```bash
scancel -u $USERNAME
```

Cancel a specific job:
```bash
scancel $JOBID
```

## Experiments

Here, we should place commands for deploying experiments with and without Slurm

To deploy all of the experiments data generation:
```
sbatch run_all.sh
```


### Train Data Augmentation Network and/or Loss Reweighting Network

**Data Augmentation Network**
```
python train_augment_net2.py --use_augment_net
```

**Loss Reweighting Network**
```
python train_augment_net2.py --use_reweighting_net --loss_weight_type=softmax
```


### Regularization Experiments

#### LSTM Experiments

The LSTM code in this repository is built on the [AWD-LSTM codebase](https://github.com/salesforce/awd-lstm-lm).
These commands should be run from inside the `rnn` folder.

First, download the PTB dataset by running:
```
./getdata.sh
```

**Tune LSTM hyperparameters with 1-step unrolling**
```
python train.py
```


### STN Comparison

To train an STN, run the following command from inside the `stn` folder:
```
python hypertrain.py --tune_all --save
```

### Train a baseline model to get a checkpoint
```
python train_checkpoint.py --dataset cifar10 --model resnet18 --data_augmentation
```

### Finetune the trained checkpoint
```
python finetune_checkpoint.py --load_checkpoint=baseline_checkpoints/cifar10_resnet18_sgdm_lr0.1_wd0.0005_aug1.pt --num_finetune_epochs=10 --wdecay=1e-4
```


### Experiment 1
Explain what experiment does, and what figure it is in the paper.

To run python script:
```
python script.py
```

To deploy with Slurm:
```
srun ...
```



# Project Structure
```
.
├── HAM_dataset.py
├── README.md
├── cutout.py
├── data_loaders.py
├── finetune_checkpoint.py
├── finetune_ift_checkpoint.py
├── grid_search.py
├── images
├── inverse_comparison.py
├── isic_config.py
├── isic_loader.py
├── kfac.py
├── kfac_utils.py
├── minst_ref.py
├── mnist_test.py
├── models
│   ├── __init__.py
│   ├── resnet.py
│   ├── resnet_cifar.py
│   ├── simple_models.py
│   ├── unet.py
│   └── wide_resnet.py
├── papers
│   ├── haoping_project
│   │   ├── main.tex
│   │   ├── neurips2019.tex
│   │   ├── neurips_2019.sty
│   │   └── references.bib
│   └── nips
│       ├── main.tex
│       ├── neurips_2019.sty
│       └── references.bib
├── random_search.py
├── requirements.txt
├── rnn
│   ├── config_scripts
│   │   ├── dropoute_ift_no_lrdecay.yaml
│   │   ├── dropouto
│   │   │   ├── dropouto_2layer_lrdecay.yaml
│   │   │   ├── dropouto_2layer_no_lrdecay.yaml
│   │   │   ├── dropouto_ift_lrdecay.yaml
│   │   │   ├── dropouto_ift_neumann_1_lrdecay.yaml
│   │   │   ├── dropouto_ift_neumann_1_no_lrdecay.yaml
│   │   │   ├── dropouto_ift_no_lrdecay.yaml
│   │   │   ├── dropouto_lrdecay.yaml
│   │   │   ├── dropouto_no_lrdecay.yaml
│   │   │   └── dropouto_perparam_ift_no_lrdecay.yaml
│   │   └── wdecay
│   │       ├── ift_wdecay_per_param_no_lrdecay.yaml
│   │       ├── wdecay_ift_lrdecay.yaml
│   │       └── wdecay_ift_neumann_1_lrdecay.yaml
│   ├── create_command_script.py
│   ├── data.py
│   ├── embed_regularize.py
│   ├── getdata.sh
│   ├── locked_dropout.py
│   ├── logger.py
│   ├── model_basic.py
│   ├── plot_utils.py
│   ├── rnn_utils.py
│   ├── run_grid_search.py
│   ├── train.py
│   ├── train2.py
│   └── weight_drop.py
├── search_configs
│   ├── cifar100_wideresnet_bern_dropout_sep.yaml
│   ├── cifar100_wideresnet_gauss_dropout_sep.yaml
│   ├── cifar10_resnet32_data_aug.yaml
│   ├── cifar10_resnet32_grid.yaml
│   ├── cifar10_resnet32_random.yaml
│   ├── cifar10_resnet32_wdecay_per_layer.yaml
│   ├── cifar10_wideresnet_bern_dropout.yaml
│   ├── cifar10_wideresnet_bern_dropout_sep.yaml
│   ├── cifar10_wideresnet_gauss_dropout.yaml
│   ├── cifar10_wideresnet_gauss_dropout_sep.yaml
│   ├── isic_grid.yaml
│   └── isic_random.yaml
├── search_scripts
│   ├── cifar100_wideresnet_bern_dropout_sep
│   ├── cifar100_wideresnet_gauss_dropout_sep
│   ├── cifar100_wideresnet_random
│   ├── cifar10_wideresnet_bern_dropout
│   ├── cifar10_wideresnet_bern_dropout_sep
│   ├── cifar10_wideresnet_gauss_dropout
│   └── cifar10_wideresnet_gauss_dropout_sep
├── srun_script.sh
├── stn
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── cifar.py
│   │   └── loaders.py
│   ├── hypermodels
│   │   ├── __init__.py
│   │   ├── alexnet.py
│   │   ├── hyperconv2d.py
│   │   ├── hyperlinear.py
│   │   └── small.py
│   ├── hypertrain.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── alexnet.py
│   │   └── small.py
│   └── util
│       ├── __init__.py
│       ├── cutout.py
│       ├── dropout.py
│       └── hyperparameter.py
├── train.py
├── train_augment_net2.py
├── train_augment_net_graph.py
├── train_augment_net_multiple.py
├── train_augment_net_slurm.py
├── train_baseline.py
├── train_checkpoint.py
└── utils
    ├── csv_logger.py
    ├── discrete_utils.py
    ├── logger.py
    ├── plot_utils.py
    └── util.py

17 directories, 103 files
```

# Authors
* **Jonathan Lorraine** - [Github](https://github.com/lorraine2)
* **Paul Vicol** - [Github](https://github.com/asteroidhouse)
* **Haoping Xu** - [Github](https://github.com/jack-xhp)

