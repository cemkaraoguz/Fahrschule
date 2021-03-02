import cv2
import numpy as np
import os, getopt, sys
from pickle import Pickler, Unpickler
from Globals import *

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
  
def getValueFromDict(indict, key, defaultVal="argrequired"):
  if key in indict.keys():
    return indict[key]
  else:
    if defaultVal=="argrequired":
      raise KeyError
    else:
      return defaultVal

def process_frame(frame, crop, size):
  obs = frame[crop[0]:crop[1], crop[2]:crop[3],:]
  if IM_CHANNELS==1:
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, size)
    obs = np.reshape(obs, [IM_CHANNELS, size[0], size[1]])
  elif IM_CHANNELS==3:
    obs = cv2.resize(obs, size)
    obs = np.transpose(obs, [2,0,1])
  else:
    raise(NotImplementedError)
  return obs/IM_DEPTH_NORM

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
  long_option_mappings = {'data_folder='    : ['data_folder', '[TRAIN] Data root directory'],
                          'cp='             : ['do_save_checkpoint', '[TRAIN] Do save checkpoint?'],
                          'cp_folder='      : ['checkpoint_folder', '[TRAIN] Checkpoint folder'],
                          'num_epochs='     : ['num_epochs', '[TRAIN] Number of epochs to train policy network'],
                          'num_epochs_vae=' : ['num_epochs_vae', '[TRAIN] Number of epochs to train VAE'],
                          'load_epoch='     : ['load_epoch', '[TRAIN] Checkpoint epoch number to load'],
                          'do_load_vae='    : ['do_load_vae', '[TRAIN] Load VAE Model from disk?'],
                          'model_file='     : ['model_file', '[TRAIN/EVAL] Path to the PN model file to be loaded'],
                          'vae_model_file=' : ['vae_model_file', '[TRAIN/EVAL] Path to the VAE model file to be loaded'],
                          'num_episode='    : ['num_episode', '[EVAL] Number of episodes to evaluate an agent'],
                          'render_mode='    : ['render_mode', '[EVAL] True for render environment on screen'],
                          'epsilon='        : ['epsilon', '[EVAL] Epsilon for epsilon-greedy action selection'], 
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