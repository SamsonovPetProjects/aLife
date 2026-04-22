import torch
from enum import IntEnum

# Настройки поля
ROWS = 20
COLS = 20
CELL_SIZE = 40

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE
FPS = 10

# Цвета
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_CELL = (0, 200, 100)
COLOR_SELECTED = (255, 100, 0)
COLOR_FOOD = (255, 50, 50)

# Множество объектов 
class Entity(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    FOOD = 3

# Платформа обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Множество действий
ACTION_MAP = {
    0: (0, 0),   # Стоять на месте
    1: (-1, 0),  # Вверх (row - 1)
    2: (1, 0),   # Вниз (row + 1)
    3: (0, -1),  # Влево (col - 1)
    4: (0, 1)    # Вправо (col + 1)
}

# Настройки Brain
MATRIX_SIZE = 7
INPUT_CHANELS = len(Entity)
NUM_ACTIONS = 5

# Настройки реплай буфера
BATCH_SIZE = 32
MEMORY_SIZE = 10000
