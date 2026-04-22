import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from config import Entity, MATRIX_SIZE, DEVICE

class Brain(nn.Module):
    def __init__(self, input_channels, num_actions, lr=0.001):
        super(Brain, self).__init__()
        self.device = DEVICE
        
        # Сверточные слои общие и для наследников
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        
        # Полносвязные слои для базовой CNN версии
        # После 7x7 -> 5x5 -> 3x3 (при kernel=3, padding=0)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # x: [Batch, Channels, 7, 7]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_step(self, states, actions, rewards, next_states, dones, gamma=0.99):
        self.train()
        
        # Текущие Q-значения
        current_q = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Целевые Q-значения Беллман - идём в max(бенефит сейчас, бенефит нового состояния) 
        with torch.no_grad():
            next_q = self.forward(next_states).max(1)[0]
            target_q = rewards + (gamma * next_q * (~dones))
            
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def split_channels(self, view):
        # Превращает матрицу обзора в многоканальный тензор для CNN
        channels = np.zeros((len(Entity), MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
        for entity in Entity:
            channels[entity.value] = (view == entity.value).astype(np.float32)
        return channels

    def save(self, filename="cnn_weights.pth"):
        base_path = os.path.dirname(__file__)
        full_path = os.path.join(base_path, filename)
        torch.save(self.state_dict(), full_path)
        print(f"--- Модель сохранена: {filename} ---")

    def load(self, filename="cnn_weights.pth"):
        base_path = os.path.dirname(__file__)
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            try:
                self.load_state_dict(torch.load(full_path, weights_only=True, map_location=self.device))
                self.eval()
                print(f"--- Модель загружена: {filename} ---")
            except EOFError:
                print(f"--- Ошибка: файл {filename} пуст или поврежден ---")
        else:
            print(f"--- Файл {filename} не найден, начинаем с нуля ---")