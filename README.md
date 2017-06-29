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

## Modules:

### 1. Agent

- QAgent : Basic deep Q learning with double Q learning

  > Human-level control through deep reinforcement learning

  > Deep Reinforcement Learning with Double Q-learning

- NStepQAgent : N step version of DQN

  > Asynchronous Methods for Deep Reinforcement Learning

- BootQAgent : Bootstrapped version of DQN

  > Deep Exploration via Bootstrapped DQN

- QACAgent : Q Actor-Critic, using Q value function as critic

  > Deterministic Policy Gradient Algorithms

- AACAgent : Advantage Actor-Critic, using R - V value function as critic

  > Deterministic Policy Gradient Algorithms

  > Asynchronous Methods for Deep Reinforcement Learning

- NStepAACAgent : N step verison of Advantage Actor-Critic

  > Asynchronous Methods for Deep Reinforcement Learning

- NFSPAgent : Neural Fictitious Self-Play

  > Deep Reinforcement Learning from Self-Play in Imperfect-Information Games

### 2. Replay

- Replay : Basic replay, randomly choose from pool and remove the oldest one

  > Human-level control through deep reinforcement learning

- PrioritizedReplay : rank base prioritized replay, choose according to err, and remove the one with least err

  > PRIORITIZED EXPERIENCE REPLAY

- ReservoirReplay : randomly choose from pool and randomly remove one, used in NFSPAgent's policy network

  > Deep Reinforcement Learning from Self-Play in Imperfect-Information Games

- TmpReplay : just for module, no replay at all

### 3. Train

- Train : normal trainer
- AsynTrain : asynchronous trainer

### 4. Env

- EnvAbstract : Env interface, similar to gym's interfaces. User has to reimplement interface functions

## TODO

1. turn python2 to python3.6
2. turn tensorflow to pytorch
3. add continue control
