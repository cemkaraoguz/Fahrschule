import numpy as np

# Training
R_TRAIN = 0.8                # Ratio of training samples in dataset

# Image
IM_WIDTH = 64                # Image dimensions
IM_HEIGHT = 64               #
IM_CHANNELS = 3              #
IM_DEPTH_NORM = 255.
IM_CROP_YMIN = 0             # Image crop area
IM_CROP_YMAX = 84            #
IM_CROP_XMIN = 0             #
IM_CROP_XMAX = 96            #

# Environment
MAX_EPISODE_LENGTH = 1000    # Maximum episode length 

# Agent
GAIN_THROTTLE = 1.0
GAIN_STEERING = 1.0
GAIN_BRAKE = 0.8
ACTION_MAPPING = {
  0: np.array([ 0.0,           0.0,           0.0       ], dtype=np.float32),  # STRAIGHT
  1: np.array([ 0.0,           GAIN_THROTTLE, 0.0       ], dtype=np.float32),  # ACCELERATE
  2: np.array([ GAIN_STEERING, 0.0,           0.0       ], dtype=np.float32),  # RIGHT
  3: np.array([ 0.0,           0.0,           GAIN_BRAKE], dtype=np.float32),  # BRAKE
  4: np.array([-GAIN_STEERING, 0.0,           0.0       ], dtype=np.float32),  # LEFT
  }
NUM_DISCRETE_ACTIONS = len(ACTION_MAPPING)      # Number of actions if discrete space is used
  