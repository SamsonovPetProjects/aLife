import random
import torch
import numpy as np

from .replay_buffer import ReplayBuffer

class SequentialBuffer(ReplayBuffer):
    def __init__(self, capacity, device, seq_len=8):
        super().__init__(capacity, device)
        self.seq_len = seq_len

    def sample(self, batch_size):
        # Нам нужно выбрать batch_size индексов так, чтобы от каждого 
        # можно было отсчитать seq_len шагов назад
        states_seq, actions, rewards, next_states, dones = [], [], [], [], []
        
        for _ in range(batch_size):
            # Выбираем случайную точку, учитывая длину последовательности
            idx = random.randint(self.seq_len, len(self.buffer) - 1)
            
            # Собираем цепочку состояний [idx - seq_len : idx]
            chunk = list(self.buffer)[idx - self.seq_len : idx]
            
            s_seq, a, r, ns, d = zip(*chunk)
            
            states_seq.append(np.array(s_seq)) # Станет [Seq, Channels, H, W]
            # Для остальных берем только ПОСЛЕДНИЙ шаг в цепочке
            actions.append(a[-1])
            rewards.append(r[-1])
            next_states.append(ns[-1])
            dones.append(d[-1])

        # Конвертируем в тензоры
        # states_seq станет [Batch, Seq, Channels, H, W]
        states = torch.as_tensor(np.array(states_seq), dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones