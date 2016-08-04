# DeepRL

Framework for deep reinforcement learning.

## Features:

- Complete algorithms are splited into modules
- Very easy to run algorithms asynchronously
- Easy to add new algorithms

## Dependences
- python2
- numpy
- zmq (for asyn train)
- tensorflow

## Modules:

### 1\. Agent

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

### 2\. Replay

- Replay : Basic replay, randomly choose from pool and remove the oldest one

  > Human-level control through deep reinforcement learning

- PrioritizedReplay : rank base prioritized replay, choose according to err, and remove the one with least err

  > PRIORITIZED EXPERIENCE REPLAY

- ReservoirReplay : randomly choose from pool and randomly remove one, used in NFSPAgent's policy network

  > Deep Reinforcement Learning from Self-Play in Imperfect-Information Games

- TmpReplay : just for module, no replay at all

### 3\. Train

- Train : normal trainer
- AsynTrain : asynchronous trainer

### 4\. Test

- Test : normal tester

### 5\. Env

- Env : Env interface, similar to gym's interfaces. User has to reimplement interface functions

## Example

<https://github.com/ppaanngggg/deep_rl_experiment>

## TODO

1. turn Chainer to Tensorflow
2. add continue control
