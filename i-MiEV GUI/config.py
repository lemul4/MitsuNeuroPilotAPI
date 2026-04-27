import os

MAX_ANGLE = 630
BUFFER_SIZE = 1024
UI_UPDATE_RATE_MS = 50
PHYSICS_UPDATE_RATE_MS = 50

# Физика
MAX_SPEED_FORWARD = 130.0
MAX_SPEED_REVERSE = -20.0
MASS_FACTOR = 0.02

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DEFAULT_ROUTE = os.path.join(PROJECT_ROOT, "data", "data_routes", "leaderboard1", "ControlLoss", "Town01_Scenario1_0.xml")
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "model_2.pth")