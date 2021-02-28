import gym
from Models import ConvVAEWrapper, ResNetWrapper
from Utils import process_frame, parseArguments, getValueFromDict
import numpy as np
import random
import os, sys
from Globals import *

def run_episode(env, model, render_mode=True, num_episode=5, seed=-1):

  reward_list = []
  t_list = []
  crop = (IM_CROP_YMIN, IM_CROP_YMAX, IM_CROP_XMIN, IM_CROP_XMAX)
  size = (IM_HEIGHT, IM_WIDTH)
  max_episode_length = 1000

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):

    obs = env.reset()
    total_reward = 0.0
    for t in range(max_episode_length):

      if render_mode:
        env.render("human")

      action = model.get_action(process_frame(obs, crop=crop, size=size))
      obs, reward, done, info = env.step(action)

      total_reward += reward

      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list
  
if __name__=="__main__":

  cmd_args = parseArguments(sys.argv[1:])
  num_episode = int(getValueFromDict(cmd_args, 'num_episode', 1))
  render_mode = str(getValueFromDict(cmd_args, 'render_mode', "True")).lower() in ["true", "1"]
  model_file = str(getValueFromDict(cmd_args, 'model_file', ""))
  if model_file=="":
    print("model_file argument is necessary!")
    sys.exit(2)
    
  checkpoint_folder, checkpoint_file = os.path.split(model_file) 

  seed = np.random.randint(10000)+10000 # Out of set of seeds used for data generation
  args = {'num_epochs': 1,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'grad_clip': 0.1,
        'steps_per_epoch': 1,
        'do_use_cuda': True,
        'num_classes': 3,
        'in_channels' : 3,
        'num_channels': 64,
        'shortcut': 'conv',
        'num_res_blocks': [3,3,3],
       }
       
  env_name = "CarRacing-v0"
  env = gym.make(env_name)
  
  policy_network = ResNetWrapper(args)
  policy_network.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)
  
  reward_list, t_list = run_episode(env, policy_network, render_mode=render_mode, num_episode=num_episode, seed=seed)
  
  print(f"Average reward for {num_episode} evaluation(s) : {np.mean(reward_list)}")
  