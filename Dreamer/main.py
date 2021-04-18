import torch 
import numpy as np
import argparse
import gym 
gym.logger.set_level(40)
import pybullet_envs
import fixed_seed

from replay_memory import ReplayBuffer
from learning import LEARNING

def main():

    # シードを設定する．
    SEED = 0
    fixed_seed.fix_seed(SEED)

    parser = argparse.ArgumentParser(description='PyTorch Dreamer')

    parser.add_argument('--buffer_capacity', type=int, default=200000, metavar='N',
                        help='Size of Replay Buffer (default: 200000)')
    parser.add_argument('--state_dim', type=int, default=30, metavar='N',
                        help='Dimension of stochastic state (default: 30)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=200, metavar='N',
                        help='Dimension of deterministic state (default: 200)')
    parser.add_argument('--model_lr', type=float, default=6e-4, metavar='G',
                        help='Learning rate of Encoder, rssm, obs_model, reward_model (default: 6e-4)')
    parser.add_argument('--value_lr', type=float, default=8e-6, metavar='G',
                        help='Learning rate of critic (default: 8e-5)') 
    parser.add_argument('--action_lr', type=float, default=8e-6, metavar='G',
                        help='Learning rate of actor (default: 8e-5)') 
    parser.add_argument('--eps', type=float, default=1e-4, metavar='G',
                        help='Eps rate (default: 1e-4)')                                     
    parser.add_argument('--seed_episodes', type=int, default=5, metavar='N',
                        help='Num of initial random episodes (default: 5)')
    parser.add_argument('--all_episodes', type=int, default=300, metavar='N',
                        help='Num of all learning episodes (default: 300)')
    parser.add_argument('--test_interval', type=int, default=10, metavar='N',
                        help='test interval (default: 10)')
    parser.add_argument('--model_save_interval', type=int, default=20, metavar='N',
                        help='model save interval (default: 20)')
    parser.add_argument('--collect_interval', type=int, default=100, metavar='N',
                        help='Num of DNN updates per one episode (default: 100)')
    parser.add_argument('--action_noise_var', type=float, default=0.3, metavar='G',
                        help='Variance of exploration noises (default: 0.3)')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='Size of batch (default: 50)')
    parser.add_argument('--chunk_length', type=int, default=50, metavar='N',
                        help='chunk size (default: 50)')
    parser.add_argument('--imagination_horizon', type=int, default=15, metavar='N',
                        help='(default: 15)') # Actor Critic の更新ために，Dreamerでどれだけ先の軌道を想像するか
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='Discount rate (default: 0.99)')
    parser.add_argument('--lambda_', type=float, default=0.95, metavar='G',
                        help='lambda parameter (default: 0.95)')
    parser.add_argument('--clip_grad_norm', type=float, default=100., metavar='G',
                        help='value of gradient clip (default: 100)')
    parser.add_argument('--free_nats', type=float, default=3., metavar='G',
                        help='(default: 3)') # KL誤差(RSSMのpriorとposteriorの間の誤差)がこの値以下の場合無視する．
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='(default: 0)')    
    parser.add_argument('--env_id', type=str, default='HalfCheetahBulletEnv-v0',
                        help='Env ID (defalut: HalfCheetah)')

    args = parser.parse_args()

    # torch.deviceを定義. この変数は後々モデルやデータをGPUに転送する時にも使います
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    learning_dreamer = LEARNING(args, device)
    learning_dreamer.run()

if __name__ == "__main__":
    main()