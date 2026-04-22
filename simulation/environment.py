import numpy as np
import random
from config import ROWS, COLS, Entity, ACTION_MAP

class Environment:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS

        self.world_map = np.zeros((self.rows, self.cols), dtype=int)
        self.food_set = set()

    def reset(self):
        # Полная очистка мира
        self.world_map.fill(Entity.EMPTY.value)
        self.food_set.clear()

    def get_empty_pos(self):
        # Ограничиваем количество попыток, чтобы избежать вечного цикла
        for _ in range(1000):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if self.world_map[r, c] == Entity.EMPTY.value:
                return r, c
        return (0, 0)

    def spawn_food(self, count):
        # +Eда
        for _ in range(count):
            r, c = self.get_empty_pos()
            self.food_set.add((r, c))
            self.world_map[r, c] = Entity.FOOD.value

    def step(self, agent, action_idx):
        # В файле конфиг есть Entity с набором действий
        dr, dc = ACTION_MAP[action_idx]
        old_r, old_c = agent.pos
        new_r, new_c = old_r + dr, old_c + dc
        
        reward = -0.01  # Базовая плата за каждый шаг
        done = False

        # Проверка границ
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # Агент ударился о границу, остается на месте
            return -1.0, False 

        target_cell = self.world_map[new_r, new_c]

        # Обработка стены
        if target_cell == Entity.WALL.value:
            return -1.0, False

        # Логика перемещения и еды
        if target_cell == Entity.FOOD.value:
            reward = 10.0
            # Гиперпараметр? 
            agent.energy = min(agent.energy + 50, agent.max_energy)
            # Удаляем еду из сета по координатам
            if (new_r, new_c) in self.food_set:
                self.food_set.remove((new_r, new_c))
        else:
            # Трата энергии за движение в пустоту
            agent.energy -= 0.5

        # Обновляем матрицу с новыми объектами
        self.world_map[old_r, old_c] = Entity.EMPTY.value
        agent.pos = (new_r, new_c)
        self.world_map[new_r, new_c] = Entity.AGENT.value

        # Проверка смерти от голода и минус вайба
        if agent.energy <= 0:
            reward = -10.0
            done = True
            self.world_map[new_r, new_c] = Entity.EMPTY.value
            
        return reward, done