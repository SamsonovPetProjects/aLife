import random
import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Выбирает случайную пачку и конвертирует её в тензоры для GPU/CPU
        # Команда, по всем вопросам логики читать ноутбук лм
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Конвертируем всё в тензоры сразу на нужный девайс
        states = torch.as_tensor(np.array(state), dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(np.array(action), dtype=torch.long).to(self.device)
        rewards = torch.as_tensor(np.array(reward), dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(np.array(done), dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)