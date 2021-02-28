import cv2
import numpy as np
import os, getopt, sys
from pickle import Pickler, Unpickler

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

def saveLogData(logdata, folder):
  filepath = os.path.join(folder, 'logdata.pkl')
  if not os.path.exists(folder):
      os.mkdir(folder)
  with open(filepath, "wb+") as f:
    Pickler(f).dump(logdata)

def loadLogData(folder):
  filepath = os.path.join(folder, 'logdata.pkl')
  if not os.path.exists(filepath):
      raise FileNotFoundError("No log data in path {}".format(filepath))
  else:
    print("Loading log data {}".format(filepath))
    with open(filepath, "rb") as f:
      return Unpickler(f).load()
      
def parseArguments(argv):
  long_option_mappings = {'data_folder=': ['data_folder', 'Data root directory'],
                          'cp=': ['do_save_checkpoint', 'Do save checkpoint?'],
                          'cp_folder=': ['checkpoint_folder', 'Checkpoint folder'],
                          'cp_step=': ['checkpoint_step', 'Checkpoint saving interval'],
                          'num_epochs=': ['num_epochs', 'Number of epochs to train'],
                          }
                          
  def print_arguments():
    print("Arguments : ")
    print("".join(["  "+k[:-1]+":"+" "*(16-len(k[:-1]))+v[1]+"\n" for k,v in long_option_mappings.items()]))

  try:
    opts, args = getopt.getopt(argv, "h", list(long_option_mappings.keys()))
  except getopt.GetoptError:
    print("Usage:")
    print_arguments()
    sys.exit(2)
  ret = {}
  for opt, arg in opts:
    if opt=='-h':
      print_arguments()
    else:      
      ret[long_option_mappings[opt[2:]+"="][0]] = arg
  return ret