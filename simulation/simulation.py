import pygame
import sys
import matplotlib.pyplot as plt

from config import (WIDTH, HEIGHT, CELL_SIZE, FPS, COLOR_BG, COLOR_GRID, COLOR_FOOD, 
                    Entity, BATCH_SIZE, MEMORY_SIZE, INPUT_CHANELS, NUM_ACTIONS, DEVICE, ROWS, COLS)

from entities.agent import Agent
from brain.model import Brain
from brain.replay_buffer import ReplayBuffer
from environment import Environment 

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Evolved Life Simulation (DQN + Replay)")
        self.clock = pygame.time.Clock()

        # Флаги управления
        self.render = True
        self.paused = False
        self.total_steps = 0
        
        # Инициализация Среды
        self.env = Environment()
        self.agents = []
        
        # Инициализация модели
        self.shared_brain = Brain(INPUT_CHANELS, NUM_ACTIONS).to(DEVICE)
        
        # Подгрузка весов
        try:
            self.shared_brain.load("cnn_weights.pth")
        except:
            print("Начинаем обучение с нуля.")
            
        self.memory = ReplayBuffer(MEMORY_SIZE, DEVICE)
        
        # 3. Состояние UI
        self.selected_agent = None
        self.epsilon = 0.15 

        # 4. Старт
        self.spawn_agents(3)
        self.env.spawn_food(15)

    def spawn_agents(self, count):
        for _ in range(count):
            r, c = self.env.get_empty_pos()
  
            new_agent = Agent(r, c, brain=self.shared_brain)
            self.agents.append(new_agent)

            self.env.world_map[r, c] = Entity.AGENT.value

    def update_logic(self):
        # МЕНЯТЬ ПО ХОДУ ОБУЧЕНИЯ
        if len(self.env.food_set) < 10:
            self.env.spawn_food(5)

        for agent in self.agents[:]:
            state_matrix = agent.get_view(self.env.world_map)
            state_tensor = self.shared_brain.split_channels(state_matrix)
            
            # Действие агента
            action_idx = agent.decide_action(state_tensor, epsilon=self.epsilon)
            
            # Шаг в физическом мире (Environment)
            reward, is_dead = self.env.step(agent, action_idx)
            
            next_state_matrix = agent.get_view(self.env.world_map)
            next_state_tensor = self.shared_brain.split_channels(next_state_matrix)
            
            # Сохранение опыта
            self.memory.push(state_tensor, action_idx, reward, next_state_tensor, is_dead)
            
            # Обучение
            if len(self.memory) > BATCH_SIZE:
                batch = self.memory.sample(BATCH_SIZE)
                self.shared_brain.train_step(*batch)
                
            if is_dead:
                self.agents.remove(agent)
                self.spawn_agents(1)

        # Автосохранение чекпоинтов (раз в настроить потом итераций цикла)
        self.total_steps += 1
        if self.total_steps % 100000 == 0:
            self.shared_brain.save(f"cnn_weights_{self.total_steps}.pth")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.shared_brain.save("cnn_weights.pth")
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_s:
                    self.shared_brain.save("cnn_weights.pth")
                if event.key == pygame.K_v:
                    self.render = not self.render
                    print(f"Графика: {'ВКЛ' if self.render else 'ВЫКЛ (Турбо)'}")
                if event.key == pygame.K_TAB:
                    self.debug_matrix(self.env.world_map)

    def draw(self):
        self.screen.fill(COLOR_BG)
        
        # Статичные объекты
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)
                
                cell_val = self.env.world_map[r, c]
                if cell_val == Entity.FOOD.value:
                    pygame.draw.circle(self.screen, COLOR_FOOD, rect.center, CELL_SIZE // 3)
                elif cell_val == Entity.WALL.value:
                    pygame.draw.rect(self.screen, (50, 50, 50), rect)

        # РДинамические объекты
        for agent in self.agents:
            # is_selected потом доделаю
            agent.draw(self.screen, is_selected=False)

        pygame.display.flip()

    def debug_matrix(self, matrix):
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar(label='Entity Type')
        plt.title(f"World Map Step: {self.total_steps}")
        plt.show()

    def run(self):
        while True:
            self.handle_events()
            
            if not self.paused:
                self.update_logic()
                
            if self.render:
                self.draw()
                self.clock.tick(FPS)
            else:
                # Дальше без фпс :)
                self.clock.tick(0)