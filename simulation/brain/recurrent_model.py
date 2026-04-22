import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from .model import Brain

class RecurrentBrain(Brain):
    def __init__(self, input_channels, num_actions, lr=0.0005):
        # Вызываем конструктор родителя, чтобы создать сверточные слои
        super(RecurrentBrain, self).__init__(input_channels, num_actions, lr)
        
        # Явно сохраняем количество действий, чтобы методы класса его видели
        self.num_actions = num_actions
        
        # Короче...
        # input_size: 32*3*3 (признаки от CNN)
        # hidden_size: 128 (размер памяти)
        self.hidden_size = 128
        self.gru = nn.GRU(input_size=32 * 3 * 3, hidden_size=self.hidden_size, batch_first=True)
        
        # Финальный слой теперь берет выход из GRU
        self.fc_final = nn.Linear(self.hidden_size, num_actions)

    def forward(self, x, hidden=None):
        # x: [Batch, Seq, Channels, H, W] или [Batch, Channels, H, W]
        
        # 1. КОНВЕРТИРУЕМ В ТЕНЗОР, ЕСЛИ ЭТО NUMPY
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Если пришел 4D тензор, добавляем размерность последовательности (Sequence Length = 1)
        if x.dim() == 4:
            batch_size, c, h, w = x.size()
            x = x.view(batch_size, 1, c, h, w)
            
        batch_size, seq_len, c, h, w = x.size()
        
        # Прогоняем каждый кадр через свертки (схлопываем Batch и Seq для скорости)
        x = x.view(batch_size * seq_len, c, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # [Batch*Seq, 32*3*3]
        
        # Возвращаем размерность Sequence для GRU
        x = x.view(batch_size, seq_len, -1) # [Batch, Seq, Features]
        
        # GRU проход
        output, hidden = self.gru(x, hidden)
        
        # Берем последний выход последовательности для принятия решения
        last_output = output[:, -1, :]
        return self.fc_final(last_output), hidden

    def get_action_with_hidden(self, x, hidden, epsilon=0.1):
        """
        Метод для принятия решения в симуляции. 
        """
        # 1. КОНВЕРТИРУЕМ В ТЕНЗОР, ЕСЛИ ЭТО NUMPY
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Эпсилон-жадная стратегия
        if random.random() < epsilon:
            self.eval()
            with torch.no_grad():
                # Теперь x точно тензор, unsqueeze сработает
                input_x = x.unsqueeze(0).unsqueeze(0).to(self.device)
                _, next_hidden = self.forward(input_x, hidden)
                return random.randint(0, self.num_actions - 1), next_hidden
        
        self.eval()
        with torch.no_grad():
            # x: [Channels, H, W] -> [1, 1, C, H, W]
            input_x = x.unsqueeze(0).unsqueeze(0).to(self.device)
            
            logits, next_hidden = self.forward(input_x, hidden)
            
            action = torch.argmax(logits).item()
            return action, next_hidden

# ЕСЛИ НЕ ЯСНА ЛОГИКА - ПИСАТЬ В ТГ, А НЕ КОММЕНТАРИИ В КОДЕ