import pickle
import os
from Utils import process_frame
from torch.utils.data import Dataset

class CarRacingDataset(Dataset):

  IM_WIDTH = 96
  IM_HEIGHT = 96
  IM_CROP_YMIN = 0
  IM_CROP_YMAX = 84
  IM_CROP_XMIN = 0
  IM_CROP_XMAX = 96
  
  def __init__(self, root, size=(IM_HEIGHT, IM_WIDTH), 
    crop=(IM_CROP_YMIN, IM_CROP_YMAX, IM_CROP_XMIN, IM_CROP_XMAX)):
    self.root = root
    self.size = size
    self.crop = crop
    self.samples = self.load_files()
    
  def __len__(self):
    return len(self.samples)
    
  def __getitem__(self, idx):
    return self.samples[idx]
        
  def load_files(self):
    data = []
    for file in os.listdir(self.root):
      if file.endswith(".pkl"):
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
    
