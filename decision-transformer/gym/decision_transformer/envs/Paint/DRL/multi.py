import cv2
import torch
import numpy as np
from ..env import Paint
from ..utils.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fastenv():
    def __init__(self,
                 max_episode_length=10, env_batch=1, \
                 writer=None):
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = Paint(self.env_batch, self.max_episode_length)
        self.env.load_data()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.writer = writer
        self.test = True
        self.log = 0
        self.begin_num = 0

    def save_image(self, model_type, eval_iter, step):

        gt = cv2.cvtColor((to_numpy(self.env.gt[0].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
        canvas = cv2.cvtColor((to_numpy(self.env.canvas[0].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
        dir = './'+model_type+'_result/iter_' + str(eval_iter) +'/'+str(self.env.id[0] % 5)+'/'

        if not os.path.exists(dir+'canvas/'):
            os.makedirs(dir+'canvas/')

        
        cv2.imwrite(dir + str(self.env.id[0] // 5) +'_'  + '_target.png', gt)
        cv2.imwrite(dir +'canvas/'+ str(self.env.id[0] // 5) +'_' + str(step) + '_canvas.png', canvas)

    def step(self, action):
        with torch.no_grad():
            ob, r, d, _ = self.env.step(torch.tensor(action).to(device))
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, _

    def get_dist(self):
        return to_numpy((((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2).mean(1).mean(1).mean(1))

    def reset(self, test=False, episode=0):
        # print('test?', test)
        self.test = test
        ob = self.env.reset(self.test, self.begin_num)
        self.begin_num += 1
        return ob
