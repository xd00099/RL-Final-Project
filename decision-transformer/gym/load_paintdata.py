import pickle
import sys

import numpy as np


def create_paint_dataset():
    # path: ['observations', 'next_observations', 'actions', 'rewards', 'terminals']
    data_path = './data/paint_1-100/'

    trajectories = []

    for i in range(100):
        filename = 'paint_mnist_'+str(i+1)+'.pickle'
        episode = {}
        with open(data_path+filename, 'rb') as f:
            trajectory = pickle.load(f)
        obss = []
        actions = []
        rewards = []
        terminal = []
        for transition in trajectory:
            obss.append(transition[0].reshape(-1))
            actions.append(transition[1].reshape(-1))
            rewards.append(transition[3])
            if transition[2] == 1:
                terminal.append(True)
            else:
                terminal.append(False)
        # print(obss[0])
        episode['observations'] = np.array(obss)
        # print(episode['observations'].shape)
        episode['actions'] = np.array(actions)
        # print(episode['actions'][0].shape)
        episode['rewards'] = np.array(rewards)
        episode['terminals'] = np.array(terminal)
        trajectories.append(episode)

    with open('./data/paint_test_train.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

        # print(obss, actions, done_idxs, stepwise_returns)

        # sys.exit()

    # actions = np.array(actions)
    # stepwise_returns = np.array(stepwise_returns)
    # done_idxs = np.array(done_idxs)
    #
    # # -- create reward-to-go dataset
    # start_index = 0
    # rtg = np.zeros_like(stepwise_returns)
    # for i in done_idxs:
    #     i = int(i)
    #     curr_traj_returns = stepwise_returns[start_index:i]
    #     for j in range(i-1, start_index-1, -1): # start from i-1
    #         rtg_j = curr_traj_returns[j-start_index:i-start_index]
    #         rtg[j] = sum(rtg_j)
    #     start_index = i
    # print('max rtg is %d' % max(rtg))
    #
    # # -- create timestep dataset
    # start_index = 0
    # timesteps = np.zeros(len(actions)+1, dtype=int)
    # for i in done_idxs:
    #     i = int(i)
    #     timesteps[start_index:i+1] = np.arange(i+1 - start_index)
    #     start_index = i+1
    # print('max timestep is %d' % max(timesteps))
    # print(max(rtg), min(rtg))
    # # print(done_idxs)
    # # print(len(obss), len(actions), len(done_idxs), len(rtg))
    #
    # return obss, actions, done_idxs, rtg, timesteps


create_paint_dataset()