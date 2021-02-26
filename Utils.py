import cv2
import numpy as np

class AverageMeter(object):

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def __repr__(self):
    return f'{self.avg:.2e}'

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
  
def getValueFromDict(indict, key, defaultVal=None):
  if key in indict.keys():
    return indict[key]
  else:
    if defaultVal is None:
      raise KeyError
    else:
      return defaultVal

def process_frame(frame, crop, size):
  obs = frame[crop[0]:crop[1], crop[2]:crop[3],:]/256.0
  obs = cv2.resize(obs, size)
  #obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  #obs = (obs * 255).round().astype(np.uint8)    
  obs = np.transpose(obs, [2,0,1])
  return obs
