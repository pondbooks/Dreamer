import utilfunc
import replay_memory
from dnn import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel, ValueModel, ActionModel
from agent import Agent

import os
import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import time

class LEARNING:

    def __init__(self, args, device):

        self.args = args
        self.env = utilfunc.make_env(self.args.seed, self.args.env_id)
        self.device = device

        self.replay_buffer = replay_memory.ReplayBuffer(\
                capacity=self.args.buffer_capacity,\
                observation_shape=self.env.observation_space.shape,\
                action_dim=self.env.action_space.shape[0])
        
        self.encoder = Encoder().to(self.device)
        self.rssm = RecurrentStateSpaceModel(\
                    self.args.state_dim,\
                    self.env.action_space.shape[0],\
                    self.args.rnn_hidden_dim).to(self.device)
        self.obs_model = ObservationModel(self.args.state_dim, self.args.rnn_hidden_dim).to(self.device)
        self.reward_model = RewardModel(self.args.state_dim, self.args.rnn_hidden_dim).to(self.device)
        self.value_model = ValueModel(self.args.state_dim, self.args.rnn_hidden_dim).to(self.device)
        self.action_model = ActionModel(self.args.state_dim, self.args.rnn_hidden_dim,\
                            self.env.action_space.shape[0]).to(self.device)

        self.model_params = (list(self.encoder.parameters()) +\
                  list(self.rssm.parameters()) +\
                  list(self.obs_model.parameters()) +\
                  list(self.reward_model.parameters()))
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=self.args.model_lr, eps=self.args.eps)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.args.value_lr, eps=self.args.eps)
        self.action_optimizer = torch.optim.Adam(self.action_model.parameters(), lr=self.args.action_lr, eps=self.args.eps)

    def run(self):

        # ランダムに探索して経験を貯める
        for initial_explorations in range(self.args.seed_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action) 
                self.replay_buffer.push(obs, action, reward, done)
                obs = next_obs

        for episode in range(self.args.seed_episodes, self.args.all_episodes):
            log_dir = 'Desktop\Dreamer\param'
            start = time.time()
            # エージェントの宣言
            policy = Agent(self.encoder, self.rssm, self.action_model)

            obs = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = policy(obs)
                # 探索のためガウスノイズを加える
                if self.env.action_space.shape[0] == 1:
                    action += np.random.normal(0, np.sqrt(self.args.action_noise_var))
                    action = np.array([action])
                else:
                    action += np.random.normal(0, np.sqrt(self.args.action_noise_var),\
                                                self.env.action_space.shape[0])
                next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(obs, action, reward, done)
                obs = next_obs
                total_reward += reward

            # 訓練時の報酬と経過時間をログとして表示
            print('episode [%4d/%4d] is collected. Total reward is %f' %\
                    (episode+1, self.args.all_episodes, total_reward))
            print('elasped time for interaction: %.2fs' % (time.time() - start))

            # DNNの更新
            start = time.time()
            for update_step in range(self.args.collect_interval):
                # -----------------------------------------------------------------
                #  encoder, rssm, obs_model, reward_modelの更新
                # -----------------------------------------------------------------
                observations, actions, rewards, _ = \
                    self.replay_buffer.sample(self.args.batch_size, self.args.chunk_length) 
                
                # 観測を前処理し, RNNを用いたPyTorchでの学習のためにTensorの次元を調整
                observations = utilfunc.preprocess_obs(observations)
                observations = torch.as_tensor(observations, device=self.device)
                observations = observations.transpose(3, 4).transpose(2, 3)
                observations = observations.transpose(0, 1)
                actions = torch.as_tensor(actions, device=self.device).transpose(0, 1)
                rewards = torch.as_tensor(rewards, device=self.device).transpose(0, 1)
                
                # 観測をエンコーダで低次元のベクトルに変換
                embedded_observations = self.encoder(\
                    observations.reshape(-1, 3, 64, 64)).view(self.args.chunk_length, self.args.batch_size, -1)

                # 低次元の状態表現を保持しておくためのTensorを定義
                states = torch.zeros(self.args.chunk_length, self.args.batch_size, self.args.state_dim, device=self.device)
                rnn_hiddens = torch.zeros(self.args.chunk_length, self.args.batch_size, self.args.rnn_hidden_dim, device=self.device)

                # 低次元の状態表現は最初はゼロ初期化
                state = torch.zeros(self.args.batch_size, self.args.state_dim, device=self.device)
                rnn_hidden = torch.zeros(self.args.batch_size, self.args.rnn_hidden_dim, device=self.device)

                # 状態予測を行ってそのロスを計算する（priorとposteriorの間のKLダイバージェンス）
                kl_loss = 0
                for l in range(self.args.chunk_length-1):
                    next_state_prior, next_state_posterior, rnn_hidden = \
                        self.rssm(state, actions[l], rnn_hidden, embedded_observations[l+1])
                    state = next_state_posterior.rsample()
                    states[l+1] = state
                    rnn_hiddens[l+1] = rnn_hidden
                    kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
                    kl_loss += kl.clamp(min=self.args.free_nats).mean()  # 原論文通り, KL誤差がfree_nats以下の時は無視
                kl_loss /= (self.args.chunk_length - 1)
        
                # states[0] and rnn_hiddens[0]はゼロ初期化なので以降では使わない
                states = states[1:]
                rnn_hiddens = rnn_hiddens[1:]

                # 観測を再構成, また, 報酬を予測
                flatten_states = states.view(-1, self.args.state_dim)
                flatten_rnn_hiddens = rnn_hiddens.view(-1, self.args.rnn_hidden_dim)
                recon_observations = self.obs_model(flatten_states, flatten_rnn_hiddens).view(self.args.chunk_length-1, self.args.batch_size, 3, 64, 64)
                predicted_rewards = self.reward_model(flatten_states, flatten_rnn_hiddens).view(self.args.chunk_length-1, self.args.batch_size, 1)

                # 観測と報酬の予測誤差を計算
                obs_loss = 0.5 * F.mse_loss(recon_observations, observations[1:], reduction='none').mean([0, 1]).sum()
                reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

                # 以上のロスを合わせて勾配降下で更新する
                model_loss = kl_loss + obs_loss + reward_loss
                self.model_optimizer.zero_grad()
                model_loss.backward()
                clip_grad_norm_(self.model_params, self.args.clip_grad_norm)
                self.model_optimizer.step()

                # ----------------------------------------------
                #  Action Model, Value　Modelの更新
                # ----------------------------------------------
                # Actor-Criticのロスで他のモデルを更新することはないので勾配の流れを一度遮断
                flatten_states = flatten_states.detach()
                flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

                # DreamerにおけるActor-Criticの更新のために, 現在のモデルを用いた
                # 数ステップ先の未来の状態予測を保持するためのTensorを用意
                imaginated_states = torch.zeros(self.args.imagination_horizon + 1,\
                                                 *flatten_states.shape,\
                                                  device=flatten_states.device)
                imaginated_rnn_hiddens = torch.zeros(self.args.imagination_horizon + 1,\
                                                        *flatten_rnn_hiddens.shape,\
                                                        device=flatten_rnn_hiddens.device)
                
                #　未来予測をして想像上の軌道を作る前に, 最初の状態としては先ほどモデルの更新で使っていた
                # リプレイバッファからサンプルされた観測データを取り込んだ上で推論した状態表現を使う
                imaginated_states[0] = flatten_states
                imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

                # open-loopで未来の状態予測を使い, 想像上の軌道を作る
                for h in range(1, self.args.imagination_horizon + 1):
                    # 行動はActionModelで決定. この行動はモデルのパラメータに対して微分可能で,
                    #　これを介してActionModelは更新される
                    actions = self.action_model(flatten_states, flatten_rnn_hiddens)
                    flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states,\
                                                                   actions,\
                                                                   flatten_rnn_hiddens)
                    flatten_states = flatten_states_prior.rsample()
                    imaginated_states[h] = flatten_states
                    imaginated_rnn_hiddens[h] = flatten_rnn_hiddens
                
                # 予測された架空の軌道に対する報酬を計算
                flatten_imaginated_states = imaginated_states.view(-1, self.args.state_dim)
                flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, self.args.rnn_hidden_dim)
                imaginated_rewards = \
                    self.reward_model(flatten_imaginated_states,\
                        flatten_imaginated_rnn_hiddens).view(self.args.imagination_horizon + 1, -1)
                imaginated_values = \
                    self.value_model(flatten_imaginated_states,\
                        flatten_imaginated_rnn_hiddens).view(self.args.imagination_horizon + 1, -1)

                # λ-returnのターゲットを計算
                lambda_target_values = utilfunc.lambda_target(imaginated_rewards, imaginated_values, self.args.gamma, self.args.lambda_)

                # TD(λ)ベースの目的関数で価値関数を更新
                # https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256
                # を見ながらバグをとったが，学習がおかしくなっている？
                # dreamer の実験をおこなっている pytorch のバージョンは 1.4.0

                value_loss = 0.5 * F.mse_loss(imaginated_values, lambda_target_values.detach())
                self.value_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)

                # 価値関数の予測した価値が大きくなるようにActionModelを更新
                # PyTorchの基本は勾配降下だが, 今回は大きくしたいので-1をかける
                action_loss = -1 * (lambda_target_values.mean())
                self.action_optimizer.zero_grad()
                action_loss.backward() # ここでエラーを起こす．

                clip_grad_norm_(self.value_model.parameters(), self.args.clip_grad_norm)
                self.value_optimizer.step()
                clip_grad_norm_(self.action_model.parameters(), self.args.clip_grad_norm)
                self.action_optimizer.step()
                

                # ログをTensorBoardに出力
                print('update_step: %3d model loss: %.5f, kl_loss: %.5f, '\
                    'obs_loss: %.5f, reward_loss: %.5f, '\
                    'value_loss: %.5f action_loss: %.5f'\
                        % (update_step + 1, model_loss.item(), kl_loss.item(),\
                            obs_loss.item(), reward_loss.item(),\
                            value_loss.item(), action_loss.item()))
                total_update_step = episode * self.args.collect_interval + update_step

            print('elasped time for update: %.2fs' % (time.time() - start))    

            # --------------------------------------------------------------
            #    テストフェーズ. 探索ノイズなしでの性能を評価する
            # --------------------------------------------------------------
            if (episode + 1) % self.args.test_interval == 0:
                policy = Agent(self.encoder, self.rssm, self.action_model)
                start = time.time()
                obs = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = policy(obs, training=False)
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward

                
                print('Total test reward at episode [%4d/%4d] is %f' %\
                        (episode+1, self.args.all_episodes, total_reward))
                print('elasped time for test: %.2fs' % (time.time() - start))

            if (episode + 1) % self.args.model_save_interval == 0:
                # 定期的に学習済みモデルのパラメータを保存する
                model_log_dir = os.path.join(log_dir, 'episode_%04d' % (episode + 1))
                os.makedirs(model_log_dir)
                torch.save(self.encoder.state_dict(), os.path.join(model_log_dir, 'encoder.pth'))
                torch.save(self.rssm.state_dict(), os.path.join(model_log_dir, 'rssm.pth'))
                torch.save(self.obs_model.state_dict(), os.path.join(model_log_dir, 'obs_model.pth'))
                torch.save(self.reward_model.state_dict(), os.path.join(model_log_dir, 'reward_model.pth'))
                torch.save(self.value_model.state_dict(), os.path.join(model_log_dir, 'value_model.pth'))
                torch.save(self.action_model.state_dict(), os.path.join(model_log_dir, 'action_model.pth'))