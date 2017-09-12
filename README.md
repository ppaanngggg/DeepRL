# DeepRL

Framework for deep reinforcement learning.

## Features:

- Algorithms are splited into modules
- Easy to run algorithms asynchronously
- Easy to add new algorithms

## Dependences

- python3.6
- numpy
- pytorch
- gym

## Install

1. `git clone https://github.com/ppaanngggg/DeepRL`
2. `pip install -e .`

## Modules:

### 1\. Agent

- DoubleDQNAgent: Basic deep Q learning with double Q learning

  > Human-level control through deep reinforcement learning

  > Deep Reinforcement Learning with Double Q-learning

- DDPGAgent: continue control by deep deterministic policy gradient

  > CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING

- PPOAgent: continue control by proximal policy optimization

  > Proximal Policy Optimization Algorithms

### 2\. Replay

- Replay: Basic replay, randomly choose from pool and remove the oldest one

  > Human-level control through deep reinforcement learning

- ReservoirReplay: randomly choose from pool and randomly remove one, used in NFSPAgent's policy network

  > Deep Reinforcement Learning from Self-Play in Imperfect-Information Games

- TmpReplay: just for module, no replay at all

### 3\. Train

- Train: normal trainer
- TrainEpoch:
- AsynTrainEpoch: it will

### 4\. Env

- EnvAbstract: Env interface, similar to gym's interfaces. User has to reimplement interface functions

## TODO

1. turn python2 to python3.6
2. turn tensorflow to pytorch
3. add more agent
4. well doc
