import random
import numpy as np
import torch
import pickle
import os
from skimage.transform import resize
from torchvision import transforms

class RaceCarDataLoader():
  
  IM_WIDTH = 96
  IM_HEIGHT = 96
  
  def __init__(self, root, batch_size, shuffle, size=(IM_HEIGHT, IM_WIDTH), crop=(0, 84, 0, 96)):
    self.root = root
    self.batch_size = batch_size
    self.size = size
    self.crop = crop
    self.database = self.load_files()
    self.num_samples = len(self.database)
    indices = list(range(self.num_samples))
    if shuffle:
      random.shuffle(indices)
    self.indices = [indices[i*batch_size:min((i+1)*batch_size, len(indices))] for i in range((self.num_samples//batch_size)+1)]
    self.tensor_transform = transforms.ToTensor()
    
  def __len__(self):
    return len(self.indices)
  
  def __iter__(self):
    for batch_indices in self.indices:
      batch_frame = []
      batch_reward = []
      batch_action = []
      for i in batch_indices:
        batch_frame.append(self.database[i][0])
        batch_reward.append(self.database[i][1])
        batch_action.append(self.database[i][2])
      yield batch_frame, batch_reward, batch_action

  def load_files(self):
    data = []
    for file in os.listdir(self.root):
      if file.endswith(".pkl"):
        data.extend(self.read_file(os.path.join(self.root, file)))
    return data
  
  def read_file(self, file_path):
    data = []
    is_first = True
    with open(file_path, 'rb') as fp:
      while True:
        try:
          sample = pickle.load(fp)
          frame = self.process_frame(sample[0])
          if not is_first:
            # Skip the first frame, we need st, at, rt+1
            data.append((frame_old, sample[1], sample[2]))
          else:
            is_first = False
          frame_old = np.array(frame) # copy
        except EOFError:
          break
    return data
    
  def process_frame(self, frame):
    obs = frame[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3],:].astype(np.float)/255.0
    obs = resize(obs, self.size)
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    #obs = (obs * 255).round().astype(np.uint8)
    obs = np.transpose(obs, [2,0,1])
    return obs
