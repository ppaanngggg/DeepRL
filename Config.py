# rl type
q_learning = 'q_learning'
actor_critic = 'actor_critic'

double_q = True
prioritized_replay = True
bootstrap = False

# bootstrap
K = 10  # head number
p = 0.5  # for mask

# randomly choose actions
epsilon = 1.
epsilon_decay = 0.99
epsilon_underline = 0.1

# prioritized experience replay
alpha = 0.7
beta = 0.5
beta_add = 0.0001

# reward decay
gamma = 0.9

# gpu
gpu = False

# config for replay
replay_N = 10000  # size of replay
replay_p = 1.0  # p for add into replay

# train
batch_size = 32

step_total = 1e8  # when to exit
step_train = 1  # when to train
setp_update_target = 10000  # when to update target
step_save = 100000  # when to save model
