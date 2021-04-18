import numpy as np
import gym
from wrapper import GymWrapper_PyBullet, RepeatAction

import torch

def make_env(SEED, env_id):
    env = gym.make(env_id)
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED) 
    env = GymWrapper_PyBullet(env, cam_dist=2, cam_pitch=0, render_width=64, render_height=64)
    env = RepeatAction(env, skip=2) 
    return env

def preprocess_obs(obs):
    # 画像の変換 int型[0, 255] から float型[-0.5, 0.5]へ
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs

# Dreamerでは価値関数の学習を行う．通常のTD誤差ではなく，TD(lambda)をベースにしたターゲット値を計算する．
def lambda_target(rewards, values, gamma, lambda_):

    V_lambda = torch.zeros_like(rewards, device=rewards.device) # 引数と同じ次元の0を要素とする変数を用意
    
    H = rewards.shape[0] - 1
    V_n = torch.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]

    for n in range(1, H+1):
        # n-step return を計算する
        # 系列が途中で終わってしまう場合は，可能な中で最大のnを用いたn-stepを使う．
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n+1):
            if k==n:
                V_n[:-n] += (gamma ** (n-1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k-1)) * rewards[k:-n+k]
        
        # lambda_ で n-stap return を重みづけてlamda returnを計算する．
        if n==H:
            V_lambda += (lambda_ ** (H-1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n-1)) * V_n

    return V_lambda

