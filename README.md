Teaching with Commentaries
==========================
This repository contains code to accompany the paper [Teaching with Commentaries](https://arxiv.org/abs/2011.03037), ICLR 2021. 

## Introduction to Commentaries
Commentaries represent learned, per-example/per-iteration/per-class meta-information that can be provided to a neural network model during training to improve training speed, performance, and provide insights. 
This repository provides an implementation of the two algorithms in the original paper (Algorithms 1 and 2) to learn commentaries through backpropagation through training (Alg 1) or implicit differentiation (Alg 2). 

## Setup 
To get started, create a conda environment with the required packages:

```
conda create -n commentaries python=3.7
conda activate commentaries
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install -r requirements.txt
```

## Algorithm 1: Backpropagation through Training for Commentary Learning
We provide an implementation of Algorithm 1 for commentary learning by backpropagation through training. We demonstrate this for commentaries that encode a per-example, per-iteration importance weight that is used to weight a network's loss during training. These weights are produced by a commentary (or teacher) neural network.

### Learning Commentaries
To learn a network that produces per-example weights with the following configuration:

- Dataset: CIFAR-10
- Number of inner optimisation steps used for student learning: 100
- Number of outer optimisation steps (meta-iterations) used for teacher learning: 100 
- Inner and outer optimisation learning rates: 1e-4

run:

```
python comms-weighting.py --dataset cifar10
```

See the script for more details on running other datasets (CIFAR-100, MNIST), varying the optimisation steps, and learning rates.

### Evaluating Commentaries
To train a new student with the learned commentaries, run: 

```
python student_train_weighting.py --dataset cifar10 --teachckpt /path/to/teacher/ckpt
```

and specify the correct teacher network checkpoint path. By passing a `--baseline` flag, this can also be run without the commentaries.

## Algorithm 2: Implicit Differentation for Commentary Learning
We provide an implementation of Algorithm 2 for commentary learning by implicit differentation. This is demonstrated to learn a label-dependent data augmentation scheme, where pairs of examples are blended together with a learned proportion determined by the two class labels.  Code for Algorithm 2 builds on the work from this repository: [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://github.com/lorraine2/implicit-hyper-opt/).

### Learning Commentaries
To learn a blending augmentation policy with the following configuration:

- Dataset: CIFAR-10
- Student model: ResNet-18
- Inner optimisation learning rate (for student network): 1e-3
- Neumann series steps: 1 
- Warmup epochs: -1 (no warmup)
- Training epochs: 100

run:

```
cd third_party/implicit_hyper_opt/
python train_mixup.py --dataset cifar10 --model resnet18 --epochs 100 --num_neumann_terms 1 --lr 1e-3 --hyperlr 1e-2
```

See `third_party/implicit_hyper_opt/aug_args_loading_utils.py` and the train script for more details on configurations.

### Evaluating Commentaries
To train a student network with the learned augmentation policy and standard LR/scheduler settings for CIFAR-10, run: 

```
cd third_party/implicit_hyper_opt/
python test_mixup.py --dataset cifar10 --model resnet18 --epochs 100 --load_checkpoint /path/to/aug/ckpt
```

and specify the correct augmentation policy checkpoiunt. Without a learned checkpoint, this will just run a standard CIFAR-10 run without blending augmentations (and with standard crop/flip augmentations for CIFAR).

## Applying commentary learning to other problems
The implementations for Alg 1 and 2 can serve as a starting point for other applications of commentary learning. By changing the relevant loss functions for inner optimisation in Alg 1 and the train/val loss functions in Alg 2, a wide variety of different commentary structures can be realised (such as auxiliary tasks, attention masks, etc). 

## Citation
If you make use of this paper or the code, please cite our work:

```
@inproceedings{
    raghu2020teaching,
    title={Teaching with Commentaries},
    author={Raghu, Aniruddh and Raghu, Maithra and Kornblith, Simon and Duvenaud, David and Hinton, Geoffrey},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=4RbdgBh9gE},
}
```

### Note: This is not an officially supported Google product.