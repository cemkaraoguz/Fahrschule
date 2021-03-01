import numpy as np

R_TRAIN = 1.0                # Ratio of training samples in dataset
IM_WIDTH = 64                # Image dimensions
IM_HEIGHT = 64               #
IM_CHANNELS = 1              #
IM_CROP_YMIN = 0             # Image crop area
#IM_CROP_YMAX = 96            #
IM_CROP_YMAX = 84           #
IM_CROP_XMIN = 0             #
IM_CROP_XMAX = 96            #
NUM_DISCRETE_ACTIONS = 5     # Number of actions if discrete space is used
MAX_EPISODE_LENGTH = 1000    # Maximum episode length 

GAIN_THROTTLE = 1.0
GAIN_STEERING = 1.0
GAIN_BRAKE = 0.8
ACTION_MAPPING = np.array([
  [ 0.0,           0.0,           0.0       ],  # STRAIGHT
  [ 0.0,           GAIN_THROTTLE, 0.0       ],  # ACCELERATE
  [ GAIN_STEERING, 0.0,           0.0       ],  # RIGHT
  [ 0.0,           0.0,           GAIN_BRAKE],  # BRAKE
  [-GAIN_STEERING, 0.0,           0.0       ],  # LEFT
  ], dtype=np.float32)