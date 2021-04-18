import numpy as np

class ReplayBuffer(object):
    # RNNに適したリプレイバッファ

    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        # indexは巡回し，もっとも古い経験を上書きする．
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        # 経験をリプレイバッファから(ほぼ一様に)サンプルする
        # (batch_size, chunk_size, 各要素の次元) の配列が返される
        # 各バッチは連続した経験になっている (chunk_lengthを大きくするとそれより前でdoneになる経験しかないと無限ループになる？)

        episode_borders = np.where(self.dones)[0] # done=True になっているindexを返す．np.where は該当する場所の座標をタプルで返す
        sampled_indexes = []
        
        for _ in range(batch_size):
            cross_border = True
            while cross_border: #サンプルの間にdone=Trueとなるものが入らないよう工夫
                initial_index = np.random.randint(len(self) - chunk_length + 1) # 最初のindexを一様に決定. len(self) はリプレイバッファの経験の現在の数
                final_index = initial_index + chunk_length - 1 # chunk_sizeの分連続した経験をサンプルするが，この場合の最後のindexを保存
                cross_border = np.logical_and(initial_index <= episode_borders,\
                                            episode_borders < final_index).any() # bool値のarrayになっている．いずれかの要素がTrueならばTrueを返す．
                                            #ここが無限ループになる可能性がある．
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(\
                        batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(\
                        batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(\
                        batch_size, chunk_length, 1)
        sampled_dones = self.dones[sampled_indexes].reshape(\
                        batch_size, chunk_length, 1)
        
        return sampled_observations, sampled_actions, sampled_rewards, sampled_dones

    def __len__(self):
        return self.capacity if self.is_filled else self.index
