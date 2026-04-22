import numpy as np
import random
import pygame
import torch
from config import MATRIX_SIZE, CELL_SIZE, Entity

class Agent:
    def __init__(self, r, c, brain):
        self.pos = (r, c)
        self.brain = brain # Пока мозг общий, но это временно
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def get_view(self, world_map):
        # Формат - матрица MATRIX_SIZE * MATRIX_SIZE
        r, c = self.pos
        half = MATRIX_SIZE // 2
        
        # Костыль стены
        padded_map = np.pad(world_map, pad_width=half, mode='constant', constant_values=Entity.WALL.value)
        
        # Вырезаем кусок матрицы обзора
        view = padded_map[r:r + MATRIX_SIZE, c:c + MATRIX_SIZE]
        return view

    def decide_action(self, state_tensor, epsilon=0.1):
        """
        Выбирает действие (Эпсилон-жадная стратегия).
        state_tensor: уже разделенный на каналы тензор [Channels, H, W]
        """
        # Случайное действие
        if random.random() < epsilon:
            return random.randint(0, 4) # Переделать на константы
            
        # Умное (не всегда) действие
        with torch.no_grad():
            # Добавляем размерность Batch, чтобы нейронке было комфортно: [1, C, H, W]
            state_batch = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.brain.device if hasattr(self.brain, 'device') else 'cpu')
            
            # Получаем Q-значения от мозга и берем максимальное
            q_values = self.brain(state_batch)
            action = torch.argmax(q_values).item()
            return action

    def draw(self, screen, is_selected):
        r, c = self.pos
        center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - 2
        pygame.draw.circle(screen, self.color, center, radius)
        
        # Обводка, если выбран (пока упрощено)
        if is_selected:
            pygame.draw.circle(screen, (255, 0, 0), center, radius + 2, 2)
            
        # Полоска энергии
        energy_ratio = max(0, self.energy / self.max_energy)
        health_rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE - 5, CELL_SIZE * energy_ratio, 3)
        pygame.draw.rect(screen, (0, 255, 0), health_rect)