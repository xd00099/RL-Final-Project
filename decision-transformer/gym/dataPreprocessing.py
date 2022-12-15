import sys

import neorl
import numpy as np

import pickle

env = neorl.make('ib')
env.reset()
state_dim = 74
act_dim = 14
max_ep_len = 1000
env_targets = [5000, 2500]
scale = 1000.
todo = [[]]
# load dataset

# dataset_path = './data/'+'citylearn'+'_'+'medium'+'_'+'1000'+'_'+'train'+'.pkl'
# with open(dataset_path, 'rb') as f:
#     trajectories = pickle.load(f)
# print(trajectories[0]['next_observations'])
# print(trajectories[1]['observations'])
# sys.exit()

type = ['high', 'medium', 'low']
for t in type:
    numbers = [10000, 1000, 100]
    for number in numbers:
        paths = []
        print('processing', t, number)
        train_data, val_data = env.get_dataset(data_type=t, train_num=number, need_val=False, val_ratio=0.2,
                                               use_data_reward=True)

        for i in range(len(train_data['index'])):
            path = {}
            if i == len(train_data['index'])-1:
                r = len(train_data['reward'])
            else:
                r = train_data['index'][i + 1]
            l = train_data['index'][i]
            # print(l, r)

            obs = train_data['obs'][l:r]
            next_obs = train_data['next_obs'][l:r]
            action = train_data['action'][l:r]
            rewards = train_data['reward'][l:r]
            terminals = train_data['done'][l:r]
            path['observations'] = np.array(obs)
            path['next_observations'] = np.array(next_obs)
            path['actions'] = np.array(action)
            path['rewards'] = np.array(rewards).reshape(-1)
            path['terminals'] = np.array(terminals).reshape(-1)
            terminals = path['terminals']
            if not (not(path['terminals'][l:r-1].any()) and path['terminals'][-1] == 1):
                print(path['terminals'])
                print('error in index. Starting from',l, 'to', r)

            # print(terminals == 1, path['terminals'][r-1:r])
            terminals = terminals == 1

            # print(terminals)
            path['terminals'] = np.array(terminals)
            # print(path['terminals'])
            print(path['observations'].shape, path['next_observations'].shape, path['actions'].shape, path['rewards'].shape,
                  path['terminals'].shape)

            paths.append(path)
            # print(paths)
            sys.exit()

        name = './data/'+'ib'+'_'+t+'_'+str(number)+'_'+'train'+'.pkl'
        with open(name, 'wb') as f:
            pickle.dump(paths, f)



# ['observations', 'next_observations', 'actions', 'rewards', 'terminals']
# ['obs', 'next_obs', 'action', 'reward', 'done', 'index']
# print(train_data['obs'][1])
# print(train_data['next_obs'][1])
# print(train_data['action'][1])
# print(train_data['reward'][1])
# print(train_data['done'][1])
# print(train_data['index'][1])
# print(len(train_data['index']), len(train_data['done']))
# # for i in range