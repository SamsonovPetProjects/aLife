import pygame
import sys
import matplotlib.pyplot as plt
from config import (WIDTH, HEIGHT, CELL_SIZE, FPS, COLOR_BG, COLOR_GRID, COLOR_FOOD, 
                    Entity, BATCH_SIZE, MEMORY_SIZE, INPUT_CHANELS, NUM_ACTIONS, DEVICE)

from entities.agent import Agent
from brain.model import Brain
from brain.recurrent_model import RecurrentBrain
from brain.replay_buffer import ReplayBuffer
from brain.sequential_buffer import SequentialBuffer
from environment import Environment 

class Simulation:
    def __init__(self, use_rnn=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Evolved Life Simulation")
        self.clock = pygame.time.Clock()

        self.use_rnn = use_rnn
        self.render = True
        self.paused = False
        self.total_steps = 0
        
        self.env = Environment()
        self.agents = []
        
        # --- ВЫБОР АРХИТЕКТУРЫ ---
        if self.use_rnn:
            print("RUNNING: RNN MODE")
            self.shared_brain = RecurrentBrain(INPUT_CHANELS, NUM_ACTIONS).to(DEVICE)
            self.memory = SequentialBuffer(MEMORY_SIZE, DEVICE, seq_len=10)
            self.weight_file = "rnn_weights.pth"
        else:
            print("RUNNING: CNN MODE")
            self.shared_brain = Brain(INPUT_CHANELS, NUM_ACTIONS).to(DEVICE)
            self.memory = ReplayBuffer(MEMORY_SIZE, DEVICE)
            self.weight_file = "cnn_weights.pth"
            
        self.shared_brain.load(self.weight_file)
        self.epsilon = 0.15 

        self.spawn_agents(3)
        self.env.spawn_food(15)

    def spawn_agents(self, count):
        for _ in range(count):
            r, c = self.env.get_empty_pos()
            new_agent = Agent(r, c, brain=self.shared_brain)
            # Инициализация пустой памяти для нового агента
            new_agent.hidden_state = None 
            self.agents.append(new_agent)
            self.env.world_map[r, c] = Entity.AGENT.value

    def update_logic(self):
        # МЕНЯТЬ ПО ХОДУ ОБУЧЕНИЯ
        if len(self.env.food_set) < 10:
            self.env.spawn_food(5)

        for agent in self.agents[:]:
            state_matrix = agent.get_view(self.env.world_map)
            state_tensor = self.shared_brain.split_channels(state_matrix)
            
            # Логика принятия решения в зависимости от типа мозга
            if self.use_rnn:
                action_idx, next_hidden = self.shared_brain.get_action_with_hidden(
                    state_tensor, agent.hidden_state, epsilon=self.epsilon
                )
                agent.hidden_state = next_hidden
            else:
                action_idx = agent.decide_action(state_tensor, epsilon=self.epsilon)
            
            reward, is_dead = self.env.step(agent, action_idx)
            
            next_state_matrix = agent.get_view(self.env.world_map)
            next_state_tensor = self.shared_brain.split_channels(next_state_matrix)
            
            self.memory.push(state_tensor, action_idx, reward, next_state_tensor, is_dead)
            
            if len(self.memory) > BATCH_SIZE:
                batch = self.memory.sample(BATCH_SIZE)
                self.shared_brain.train_step(*batch)
                
            if is_dead:
                self.agents.remove(agent)
                self.spawn_agents(1)

        self.total_steps += 1
        if self.total_steps % 100000 == 0:
            save_name = f"{'rnn' if self.use_rnn else 'cnn'}_weights_{self.total_steps}.pth"
            self.shared_brain.save(save_name)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.shared_brain.save(self.weight_file)
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self.render = not self.render
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def draw(self):
        self.screen.fill(COLOR_BG)
        # Отрисовка сетки
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)
                
                cell_val = self.env.world_map[r, c]
                if cell_val == Entity.FOOD.value:
                    pygame.draw.circle(self.screen, COLOR_FOOD, rect.center, CELL_SIZE // 3)

        # РДинамические объекты
        for agent in self.agents:
            agent.draw(self.screen, is_selected=False)

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            if not self.paused:
                self.update_logic()
            if self.render:
                self.draw()
                self.clock.tick(FPS)
            else:
                self.clock.tick(0) # Дальше без фпс :)