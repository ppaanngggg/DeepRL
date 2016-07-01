# gpu
gpu = False

# rl type
q_learning = 'q_learning'
actor_critic = 'actor_critic'

double_q = False
prioritized_replay = False
bootstrap = True

# randomly choose actions
epsilon = 1.
epsilon_decay = 0.99
epsilon_underline = 0.1

# bootstrap
K = 10  # head number
p = 0.5  # for mask

# prioritized experience replay
alpha = 0.7
beta = 0.5
beta_add = 0.0001

# grad clip, 0 won't clip
grad_clip = 1

# reward decay
gamma = 0.9

# config for replay
replay_N = 10000  # size of replay
replay_p = 1.0  # p for add into replay

# train batch, fetch how many tuple from replay_tuple
batch_size = 32

# for trainer to count step
step_total = 1e8  # when to exit
step_train = 1  # when to train
setp_update_target = 10000  # when to update target
step_save = 10000  # when to save model
