import pickle
import os
from Utils import process_frame
from torch.utils.data import Dataset
from Globals import *
import numpy as np

class CarRacingDataset(Dataset):
  
  def __init__(self, root, size=(IM_HEIGHT, IM_WIDTH), 
    crop=(IM_CROP_YMIN, IM_CROP_YMAX, IM_CROP_XMIN, IM_CROP_XMAX),
    num_skip_frames=50, action_space="discrete"):
    self.root = root
    self.size = size
    self.crop = crop
    self.num_skip_frames = num_skip_frames
    self.action_space = action_space
    if action_space=="discrete":
      self.process_action_vector = self._process_discrete_action
    elif action_space=="continuous":
      self.process_action_vector = self._process_continuous_action
    else:
      raise(NotImplementedError)
    self.action_histogram = np.zeros(NUM_DISCRETE_ACTIONS)
    self.samples = self._load_files()
    
  def __len__(self):
    return len(self.samples)
    
  def __getitem__(self, idx):
    return self.samples[idx]
        
  def _load_files(self):
    data = []
    for file in os.listdir(self.root):
      if file.endswith(".pkl"):
        data.extend(self._read_file(os.path.join(self.root, file)))
    return data

  def _read_file(self, file_path):
    data_all = []
    with open(file_path, 'rb') as fp:
      data = pickle.load(fp)
      for i, sample in enumerate(data):
        if i<self.num_skip_frames:
          continue
        frame = process_frame(sample[0], self.crop, self.size)
        action = self.process_action_vector(sample[2])
        reward = sample[1]
        
        do_include_sample = True
        '''
        # For balancing dataset, not necessary if weights are used in the loss function
        if(action_idx==0):
          if np.random.rand()<0.2:
            do_include_sample = True
          else:
            do_include_sample = False
        
        elif(action_idx==4):
          if np.random.rand()<0.6:
            do_include_sample = True
          else:
            do_include_sample = False
        '''
        if do_include_sample:
          data_all.append((frame, reward, action))
          if self.action_space=="discrete":
            self.action_histogram[action]+=1
          
    return data_all
    
  def _process_discrete_action(self, action_vector):
    if action_vector[0]>0:
      action_idx = 2
    elif action_vector[0]<0:
      action_idx = 4
    elif action_vector[1]>0:
      action_idx = 1
    elif action_vector[2]>0:
      action_idx = 3
    else:
      action_idx = 0
    return action_idx
    
  def _process_continuous_action(self, action_vector):
    return action_vector
    
  def calculate_weights(self):
    if self.action_space!="discrete":
      raise(NotImplementedError)
    weights = self.action_histogram/np.sum(self.action_histogram)
    return 1.0-weights
      
