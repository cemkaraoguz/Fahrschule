import random
import numpy as np
import torch
import pickle
import os
from torchvision import transforms
from Utils import process_frame

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
    for idx_batch in range(len(self.indices)):
      batch_frame, batch_reward, batch_action = self.get_batch(idx_batch)
      yield batch_frame, batch_reward, batch_action 
  
  def get_batch(self, idx_batch):
    batch_frame = []
    batch_reward = []
    batch_action = []
    batch_indices = self.indices[idx_batch]
    for i in batch_indices:
      batch_frame.append(self.database[i][0])
      batch_reward.append(self.database[i][1])
      batch_action.append(self.database[i][2])
    return batch_frame, batch_reward, batch_action
  
  def load_files(self):
    data = []
    for file in os.listdir(self.root):
      if file.endswith(".pkl"):
        print("Reading ", file)
        data.extend(self.read_file(os.path.join(self.root, file)))
    return data

  def read_file(self, file_path):
    data_all = []
    with open(file_path, 'rb') as fp:
      data = pickle.load(fp)
      for sample in data:
        frame = process_frame(sample[0], self.crop, self.size)
        data_all.append((frame, sample[1], sample[2]))
    return data_all
    