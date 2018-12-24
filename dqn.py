# extending on code from
# https://github.com/58402140/Fruit
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


GRID_SIZE = 10
TARGET_UPDATE = 10
EVALUATE_EVERY = 50
N_EVALUATIONS = 5
PRINT_EVERY = 1
N_ENSEMBLE = 5
PRIOR_SCALE = 10.
N_EPOCHS = 1000
BATCH_SIZE = 128
EPSILON = .0
GAMMA = .8
CLIP_GRAD = 10
ADAM_LEARNING_RATE = 1E-4

random_state = np.random.RandomState(11)

def save_img():
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    frame = 0
    while True:
        screen = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.savefig('images/%03i.png' % frame)
        frame += 1

def episode():
    """
    Coroutine of episode.

    Action has to be explicitly send to this coroutine.
    """
    x, y, z = (
        random_state.randint(0, GRID_SIZE),  # X of fruit
        0,  # Y of dot
        random_state.randint(1, GRID_SIZE - 1)  # X of basket
    )
    while True:
        X = np.zeros((GRID_SIZE, GRID_SIZE))  # Reset grid
        X = X.astype("float32")
        X[y, x] = 1.  # Draw fruit
        bar = range(z - 1, z + 2)
        X[-1, bar] = 1.  # Draw basket

        # End of game is known when fruit is at penultimate line of grid.
        # End represents either a win or a loss
        end = int(y >= GRID_SIZE - 2)
        if end and x not in bar:
            end *= -1

        action = yield X[np.newaxis], end
        if end:
            break

        z = min(max(z + action, 1), GRID_SIZE - 2)
        y += 1


def experience_replay(batch_size, max_size=10000):
    """
    Coroutine of experience replay.

    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    memory = []
    while True:
        inds = np.arange(len(memory))
        experience = yield [memory[i] for i in random_state.choice(inds, size=batch_size, replace=True)] if batch_size <= len(memory) else None
        # send None to just get random experiences
        if experience is not None:
            memory.append(experience)
            if len(memory) > max_size:
                memory.pop(0)


class CoreNet(nn.Module):
    def __init__(self):
        super(CoreNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 10 * 10 * 16)
        return x


class HeadNet(nn.Module):
    def __init__(self):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(10 * 10 * 16, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet()
        self.net_list = nn.ModuleList([HeadNet() for k in range(n_ensemble)])

    def forward(self, x):
        return [net(self.core_net(x)) for net in self.net_list]


class NetWithPrior(nn.Module):
    def __init__(self, net, prior, prior_scale=1.):
        super(NetWithPrior, self).__init__()
        self.net = net
        self.prior = prior
        self.prior_scale = prior_scale

    def forward(self, x):
        if hasattr(self.net, "net_list"):
            return [self.net(x)[k] + self.prior_scale * self.prior(x)[k].detach() for k in range(len(self.net.net_list))]
        else:
            return self.net(x) + self.prior_scale * self.prior(x).detach()

# Recipe of deep reinforcement learning model
prior_net = EnsembleNet(N_ENSEMBLE)
policy_net = EnsembleNet(N_ENSEMBLE)
policy_net = NetWithPrior(policy_net, prior_net, PRIOR_SCALE)

opt = optim.Adam(policy_net.parameters(), lr=ADAM_LEARNING_RATE)

target_net = EnsembleNet(N_ENSEMBLE)
target_net = NetWithPrior(target_net, prior_net, PRIOR_SCALE)

exp_replays = [experience_replay(BATCH_SIZE) for k in range(N_ENSEMBLE)]
[e.next() for e in exp_replays] # Start experience-replay coroutines

for i in range(N_EPOCHS):
    ep = episode()
    S, won = ep.next()  # Start coroutine of single entire episode
    epoch_losses = [0. for k in range(N_ENSEMBLE)]
    epoch_steps = [1. for k in range(N_ENSEMBLE)]
    heads = list(range(N_ENSEMBLE))
    random_state.shuffle(heads)
    active_head = heads[0]
    try:
        policy_net.train()
        while True:
            if random_state.rand() < EPSILON:
                action = random_state.randint(-1, 2)
            else:
                # Get the index of the maximum q-value of the model.
                # Subtract one because actions are either -1, 0, or 1
                action = np.argmax(policy_net(torch.Tensor(S[None]))[active_head].data.numpy(), axis=-1)[0] - 1

            S_prime, won = ep.send(action)
            ongoing_flag = 1.
            experience = (S, action, won, S_prime, ongoing_flag)
            S = S_prime
            for k in range(N_ENSEMBLE):
                exp_replay = exp_replays[k]
                # reset batch to default None, since we are looping
                batch = None
                # .5 probability of adding to each buffer, see paper for details
                if random_state.rand() < .5:
                    continue
                else:
                    batch = exp_replay.send(experience)
                if batch:
                    inputs = []
                    actions = []
                    rewards = []
                    nexts = []
                    ongoing_flags = []
                    for s, a, r, s_prime, ongoing_flag in batch:
                        rewards.append(r)
                        inputs.append(s)
                        actions.append(a)
                        nexts.append(s_prime)
                        ongoing_flags.append(ongoing_flag)
                    Qs = policy_net(torch.Tensor(inputs))[k]
                    Qs = Qs.gather(1, torch.LongTensor(np.array(actions)[:, None].astype("int32") + 1))

                    next_Qs = target_net(torch.Tensor(nexts))[k].detach()
                    # standard Q
                    #next_max_Qs = next_Qs.max(1)[0]

                    # double Q
                    policy_next_Qs = policy_net(torch.Tensor(nexts))[k].detach()
                    policy_actions = policy_next_Qs.max(1)[1][:, None]
                    next_max_Qs = next_Qs.gather(1, policy_actions)
                    # mask based on if it is end of episode or not
                    next_max_Qs = torch.Tensor(ongoing_flags) * next_max_Qs
                    target_Qs = torch.Tensor(np.array(rewards).astype("float32")) + GAMMA * next_max_Qs

                    loss = torch.mean((Qs - target_Qs) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    #print("policy", sum([p.abs().sum().data.numpy() for p in policy_net.parameters()]))
                    #print("target", sum([p.abs().sum().data.numpy() for p in target_net.parameters()]))
                    torch.nn.utils.clip_grad_value_(policy_net.parameters(), CLIP_GRAD)
                    opt.step()
                    epoch_losses[k] += loss.detach().cpu().numpy()
                    epoch_steps[k] += 1.
    except StopIteration:
        # add the end of episode experience
        ongoing_flag = 0.
        # just put in S, since it will get masked anyways
        experience = (S, action, won, S, ongoing_flag)
        exp_replay.send(experience)

    if i % TARGET_UPDATE == 0:
        print("Updating target network at {}".format(i))
        target_net.load_state_dict(policy_net.state_dict())

    if i % PRINT_EVERY == 0:
        print("Epoch {}, head {}, loss: {}".format(i + 1, active_head, [epoch_losses[k] / float(epoch_steps[k]) for k in range(N_ENSEMBLE)]))

    if i % EVALUATE_EVERY == 0 or i == (N_EPOCHS - 1):
        #img_saver = save_img()
        #img_saver.next()
        ensemble_rewards = []
        for k in range(N_ENSEMBLE):
            avg_rewards = []
            for _ in range(N_EVALUATIONS):
                g = episode()
                S, won = g.next()
                #img_saver.send(S)
                episode_rewards = [won]
                try:
                    policy_net.eval()
                    while True:
                        act = np.argmax(policy_net(torch.Tensor(S[np.newaxis]))[k].data.numpy(), axis=-1)[0] - 1
                        S, won = g.send(act)
                        episode_rewards.append(won)
                        #img_saver.send(S)
                except StopIteration:
                    avg_rewards.append(np.sum(episode_rewards))
            ensemble_rewards.append(np.mean(avg_rewards))
        print("rewards", ensemble_rewards)
